"""Diagnóstico do E2: por que L1/L2 pioraram a validação final com 20 mil exemplos?

Hipótese: o embedding do município não é penalizado. Com os pesos das camadas
lineares encolhidos pela penalidade, a rede pode compensar aumentando os
vetores do embedding e passar a decorar médias de municípios com poucos
exemplos (~11 por município na amostra de 20 mil). Teste: repetir as variantes
sem embedding (emb_dim = 0).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enem.pipeline import load_encoded, load_prep, subsample
from enem.runs import save_result
from enem.train import TrainConfig, fit

prep = load_prep()
train = subsample(load_encoded("train"), 20_000)
val = subsample(load_encoded("val"), 100_000)
BASE = dict(hidden=[512] * 4, lr=1e-3, patience=None, scheduler="constant", batch_size=256, max_epochs=120, emb_dim=0)
for vname, overrides in {"sem_regularizacao": {}, "l2_1e-3": {"l2": 1e-3}, "l1_1e-4": {"l1": 1e-4}}.items():
    res = fit(TrainConfig(**{**BASE, **overrides}), prep, train, val)
    save_result(res, "diagnostico_l2", f"n20k_sem_emb_{vname}", {"variante": vname, "emb_dim": 0})
    h = res["history"]
    print(f"sem embedding · {vname:18s} best_val={min(x['val_rmse'] for x in h):.2f} "
          f"final train/val={h[-1]['train_rmse']:.2f}/{h[-1]['val_rmse']:.2f}", flush=True)

# Hipótese 2: penalidade L2 somada à perda ("acoplada") perde efeito no Adam,
# porque o passo adaptativo normaliza o gradiente da penalidade. Teste: weight
# decay desacoplado (AdamW), com e sem embedding.
for emb in (16, 0):
    for wd in (1e-2, 1e-1, 1.0):
        cfg = TrainConfig(**{**BASE, "emb_dim": emb, "weight_decay": wd})
        res = fit(cfg, prep, train, val)
        name = f"n20k_emb{emb}_adamw_wd{wd:g}"
        save_result(res, "diagnostico_l2", name, {"variante": f"adamw_wd{wd:g}", "emb_dim": emb})
        h = res["history"]
        print(f"emb={emb:2d} · AdamW wd={wd:<5g} best_val={min(x['val_rmse'] for x in h):.2f} "
              f"final train/val={h[-1]['train_rmse']:.2f}/{h[-1]['val_rmse']:.2f}", flush=True)
