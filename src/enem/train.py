"""Laço de treino com early stopping, penalidades L1/L2 e registro de custo."""
import copy
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .features import Encoded, Preprocessor
from .model import EnemMLP, count_params


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class TrainConfig:
    hidden: list = field(default_factory=lambda: [256, 128])
    emb_dim: int = 16
    activation: str = "relu"
    dropout: float = 0.0
    norm: str = "none"            # none | batch | layer
    l1: float = 0.0
    l2: float = 0.0
    weight_decay: float = 0.0     # decaimento desacoplado (AdamW); 0 => Adam puro
    lr: float = 1e-3
    batch_size: int = 2048
    max_epochs: int = 40
    patience: int | None = 5      # None => sem early stopping (treina todas as épocas)
    scheduler: str = "plateau"    # plateau (reduz LR quando a validação estagna) | constant
    seed: int = 42


class DeviceData:
    """Conjunto inteiro já na GPU (cabe na memória unificada do M2)."""

    def __init__(self, enc: Encoded, device):
        self.cat = torch.from_numpy(enc.cat.astype(np.int32)).to(device)
        self.num = torch.from_numpy(enc.num).to(device)
        self.muni = torch.from_numpy(enc.muni).to(device)
        self.y = torch.from_numpy(enc.y).to(device)

    def __len__(self):
        return len(self.y)

    def batch(self, idx):
        return self.cat[idx], self.num[idx], self.muni[idx], self.y[idx]


@torch.no_grad()
def predict(model, data: DeviceData, batch_size=65536) -> torch.Tensor:
    model.eval()
    out = []
    for start in range(0, len(data), batch_size):
        sl = slice(start, start + batch_size)
        out.append(model(data.cat[sl], data.num[sl], data.muni[sl]))
    return torch.cat(out)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / ss_tot,
            "bias": float(np.mean(err))}


def evaluate(model, data: DeviceData, prep: Preprocessor) -> dict:
    pred = predict(model, data).cpu().numpy()
    return regression_metrics(prep.inverse_y(data.y.cpu().numpy()), prep.inverse_y(pred))


def build_model(cfg: TrainConfig, prep: Preprocessor) -> EnemMLP:
    return EnemMLP(prep.n_onehot_, prep.n_num_, prep.n_muni_, list(cfg.hidden),
                   emb_dim=cfg.emb_dim, activation=cfg.activation,
                   dropout=cfg.dropout, norm=cfg.norm)


def fit(cfg: TrainConfig, prep: Preprocessor, train: Encoded, val: Encoded,
        trial=None, train_eval_n: int = 100_000, verbose: bool = False,
        device=None) -> dict:
    """Treina um modelo e devolve histórico, métricas e custo.

    - Métricas de treino por época são calculadas em modo eval numa amostra fixa
      do treino, para que dropout/BN não distorçam a comparação treino x validação.
    - Com `trial` (Optuna), reporta o RMSE de validação a cada época e permite poda.
    """
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    tr = DeviceData(train, device)
    va = DeviceData(val, device)
    sample = rng.choice(len(train), size=min(train_eval_n, len(train)), replace=False)
    tr_eval = DeviceData(train.subset(np.sort(sample)), device)

    model = build_model(cfg, prep).to(device)
    if cfg.weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    loss_fn = torch.nn.MSELoss()

    history = []
    best = {"val_rmse": float("inf"), "epoch": -1, "state": None}
    bad_epochs = 0
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    _sync(device)
    t0 = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(tr), generator=gen).to(device)
        for start in range(0, len(tr), cfg.batch_size):
            cat, num, muni, y = tr.batch(perm[start:start + cfg.batch_size])
            if len(y) < 2:   # BatchNorm não aceita lote de 1
                continue
            loss = loss_fn(model(cat, num, muni), y)
            if cfg.l1 > 0 or cfg.l2 > 0:
                loss = loss + model.penalty(cfg.l1, cfg.l2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        m_tr = evaluate(model, tr_eval, prep)
        m_va = evaluate(model, va, prep)
        _sync(device)
        history.append({"epoch": epoch, "train_rmse": m_tr["rmse"], "val_rmse": m_va["rmse"],
                        "train_mae": m_tr["mae"], "val_mae": m_va["mae"],
                        "train_r2": m_tr["r2"], "val_r2": m_va["r2"],
                        "lr": opt.param_groups[0]["lr"],
                        "elapsed_s": time.perf_counter() - t0})
        if verbose:
            h = history[-1]
            print(f"ep {epoch:3d}  train {h['train_rmse']:.2f}  val {h['val_rmse']:.2f}  "
                  f"r2 {h['val_r2']:.4f}  lr {h['lr']:.1e}  {h['elapsed_s']:.0f}s", flush=True)

        if cfg.scheduler == "plateau":
            sched.step(m_va["rmse"])
        if m_va["rmse"] < best["val_rmse"] - 1e-3:
            best = {"val_rmse": m_va["rmse"], "epoch": epoch,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if trial is not None:
            import optuna
            trial.report(m_va["rmse"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if cfg.patience is not None and bad_epochs >= cfg.patience:
            break

    _sync(device)
    train_time = time.perf_counter() - t0
    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if cfg.patience is not None and best["state"] is not None:
        model.load_state_dict(best["state"])   # restaura o melhor ponto (early stopping)

    # tempo de inferência em 100 mil exemplos
    n_inf = min(100_000, len(va))
    inf_data = DeviceData(val.subset(np.arange(n_inf)), device)
    predict(model, inf_data)  # aquecimento
    _sync(device)
    t1 = time.perf_counter()
    predict(model, inf_data)
    _sync(device)
    infer_time = time.perf_counter() - t1

    return {
        "config": asdict(cfg),
        "n_params": count_params(model),
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "train_time_s": train_time,
        "time_per_epoch_s": train_time / max(len(history), 1),
        "infer_time_100k_s": infer_time,
        "history": history,
        "train": evaluate(model, tr_eval, prep),
        "val": evaluate(model, va, prep),
        "model": model,
        "final_state": final_state,
    }
