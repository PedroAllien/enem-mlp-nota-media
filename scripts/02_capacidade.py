"""Experimento 1: efeito da profundidade e da largura da MLP.

Grade: profundidade {0, 1, 2, 3, 4, 6} x largura {32, 128, 512}
(profundidade 0 = regressão linear sobre as mesmas entradas).

Rodamos em duas escalas de treino para separar "capacidade" de "quantidade de
dados": com 20 mil exemplos a rede grande tende a decorar; com 500 mil o ruído
irredutível domina. Sem regularização e com LR constante, treinando todas as
épocas, para que o overfitting apareça nas curvas (o melhor ponto da validação
é registrado mesmo assim).
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
DEPTHS = [0, 1, 2, 3, 4, 6]
WIDTHS = [32, 128, 512]

prep = load_prep()
train_full = load_encoded("train")
val = subsample(load_encoded("val"), 100_000)
only = sys.argv[1:]  # opcional: rodar só uma escala

for scale, s in SCALES.items():
    if only and scale not in only:
        continue
    train = subsample(train_full, s["n"])
    for depth in DEPTHS:
        for width in (WIDTHS if depth > 0 else [0]):
            name = f"{scale}_d{depth}_w{width}"
            cfg = TrainConfig(hidden=[width] * depth, lr=1e-3, batch_size=s["batch_size"],
                              max_epochs=s["max_epochs"], patience=None, scheduler="constant")
            res = fit(cfg, prep, train, val)
            save_result(res, "capacidade", name,
                        {"escala": scale, "n_treino": len(train), "profundidade": depth, "largura": width})
            print(f"{name:18s} params={res['n_params']:>9,d}  best_val={min(h['val_rmse'] for h in res['history']):.2f} "
                  f"final train/val={res['history'][-1]['train_rmse']:.2f}/{res['history'][-1]['val_rmse']:.2f} "
                  f"t={res['train_time_s']:.0f}s", flush=True)
