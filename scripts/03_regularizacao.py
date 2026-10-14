"""Experimento 2: estratégias de regularização numa MLP de alta capacidade.

Arquitetura fixa 4 x 512 (a maior da grade de capacidade que ainda treina
rápido). Cada variante muda UMA coisa em relação à "sem regularização".
Duas escalas de treino, como no experimento de capacidade.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enem.pipeline import load_encoded, load_prep, subsample
from enem.runs import save_result
from enem.train import TrainConfig, fit

SCALES = {
    "n20k": dict(n=20_000, batch_size=256, max_epochs=120),
    "n500k": dict(n=500_000, batch_size=2048, max_epochs=30),
}
BASE = dict(hidden=[512] * 4, lr=1e-3, patience=None, scheduler="constant")
VARIANTS = {
    "sem_regularizacao": {},
    "early_stopping": {"patience": 10},
    "dropout_0.2": {"dropout": 0.2},
    "dropout_0.5": {"dropout": 0.5},
    "l2_1e-4": {"l2": 1e-4},
    "l2_1e-3": {"l2": 1e-3},
    "l1_1e-5": {"l1": 1e-5},
    "l1_1e-4": {"l1": 1e-4},
    "batchnorm": {"norm": "batch"},
    "layernorm": {"norm": "layer"},
    "dropout_l2_es": {"dropout": 0.3, "l2": 1e-4, "patience": 10},
}

prep = load_prep()
train_full = load_encoded("train")
val = subsample(load_encoded("val"), 100_000)
only = sys.argv[1:]

for scale, s in SCALES.items():
    if only and scale not in only:
        continue
    train = subsample(train_full, s["n"])
    for vname, overrides in VARIANTS.items():
        cfg = TrainConfig(**{**BASE, "batch_size": s["batch_size"], "max_epochs": s["max_epochs"], **overrides})
        res = fit(cfg, prep, train, val)
        name = f"{scale}_{vname}"
        save_result(res, "regularizacao", name, {"escala": scale, "n_treino": len(train), "variante": vname})
        h = res["history"]
        print(f"{name:28s} best_val={min(x['val_rmse'] for x in h):.2f} (ep {res['best_epoch']}) "
              f"final train/val={h[-1]['train_rmse']:.2f}/{h[-1]['val_rmse']:.2f} "
              f"restaurado val={res['val']['rmse']:.2f} t={res['train_time_s']:.0f}s", flush=True)
