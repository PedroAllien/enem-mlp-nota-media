"""Experimento 3: busca de hiperparâmetros com Optuna (TPE + poda por mediana).

- Métrica otimizada: RMSE de validação (pontos), melhor época de cada trial.
- Dados: 400 mil exemplos de treino (amostra fixa) e 100 mil de validação.
  A amostra reduz o custo de cada trial em ~5x; os modelos finais são
  retreinados no treino completo (05_modelos_finais.py).
- O estudo é gravado em SQLite (outputs/optuna.db), então pode ser retomado
  e serve de registro auditável da busca.
"""
import json
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enem import config as C
from enem.pipeline import load_encoded, load_prep, subsample
from enem.runs import save_result
from enem.train import TrainConfig, fit

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
STUDY = "enem_mlp"
STORAGE = f"sqlite:///{C.OUTPUTS / 'optuna.db'}"

prep = load_prep()
train = subsample(load_encoded("train"), 400_000)
val = subsample(load_encoded("val"), 100_000)


def suggest(trial: optuna.Trial) -> TrainConfig:
    n_layers = trial.suggest_int("n_camadas", 1, 6)
    width = trial.suggest_int("largura", 32, 1024, log=True)
    shape = trial.suggest_categorical("formato", ["constante", "funil"])
    # funil: cada camada tem metade da largura da anterior (mín. 16)
    hidden = [max(16, width // (2 ** i)) if shape == "funil" else width for i in range(n_layers)]
    return TrainConfig(
        hidden=hidden,
        activation=trial.suggest_categorical("ativacao", ["relu", "gelu", "silu"]),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        norm=trial.suggest_categorical("normalizacao", ["none", "batch", "layer"]),
        # ampliado de [1e-7, 1e-3] para [1e-6, 1e-2] depois que E2 (500 mil) mostrou o
        # melhor λ no limite superior do intervalo original
        l2=trial.suggest_float("l2", 1e-6, 1e-2, log=True),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
        emb_dim=trial.suggest_categorical("emb_municipio", [0, 4, 8, 16, 32]),
        max_epochs=25, patience=4, scheduler="plateau",
    )


def objective(trial: optuna.Trial) -> float:
    cfg = suggest(trial)
    res = fit(cfg, prep, train, val, trial=trial)
    trial.set_user_attr("n_params", res["n_params"])
    trial.set_user_attr("train_time_s", res["train_time_s"])
    trial.set_user_attr("epochs_run", res["epochs_run"])
    trial.set_user_attr("best_epoch", res["best_epoch"])
    trial.set_user_attr("train_rmse", res["train"]["rmse"])
    trial.set_user_attr("val_r2", res["val"]["r2"])
    trial.set_user_attr("hidden", cfg.hidden)
    save_result(res, "optuna", f"trial_{trial.number:03d}", {"params": trial.params})
    return res["val"]["rmse"]


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY, storage=STORAGE, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=12, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4),
    )
    # ponto de partida conhecido: configuração "razoável" escolhida à mão
    if len(study.trials) == 0:
        study.enqueue_trial({"n_camadas": 3, "largura": 256, "formato": "constante", "ativacao": "relu",
                             "dropout": 0.1, "normalizacao": "none", "l2": 1e-6, "lr": 1e-3,
                             "batch_size": 2048, "emb_municipio": 16})
    done = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, N_TRIALS - done), gc_after_trial=True)

    df = study.trials_dataframe()
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TAB_DIR / "optuna_trials.csv", index=False)
    print("melhor trial:", study.best_trial.number, study.best_value)
    print(json.dumps(study.best_params, indent=1))
