"""Experimento 4: retreino das arquiteturas selecionadas no treino completo e
avaliação final em teste-2022 (mesmo ano) e teste-2023 (ano seguinte).

Seleção a partir do estudo Optuna (critério definido ANTES de olhar o teste):
  A) melhor trial (menor RMSE de validação);
  B) compacta: menor nº de parâmetros entre os trials a até 0,5% do melhor RMSE;
  C) contraste de profundidade: melhor trial com profundidade do "outro lado"
     (se A tem <= 3 camadas, a melhor com >= 4; senão, a melhor com <= 2).
Referências: média do treino (preditor constante) e regressão linear (0 camadas ocultas).

Cada configuração é treinada com 3 sementes; o teste usa todas as sementes
(média ± desvio) e as predições da semente 0 são salvas para análise de erros.
"""
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enem import config as C
from enem.pipeline import load_encoded, load_prep
from enem.runs import save_result
from enem.train import DeviceData, TrainConfig, evaluate, fit, get_device, predict, regression_metrics

SEEDS = [0, 1, 2]
STORAGE = f"sqlite:///{C.OUTPUTS / 'optuna.db'}"


def trial_to_config(t: optuna.trial.FrozenTrial) -> TrainConfig:
    p = t.params
    return TrainConfig(hidden=t.user_attrs["hidden"], activation=p["ativacao"], dropout=p["dropout"],
                       norm=p["normalizacao"], l2=p["l2"], lr=p["lr"], batch_size=p["batch_size"],
                       emb_dim=p["emb_municipio"])


def select_trials(study: optuna.Study) -> dict:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = min(done, key=lambda t: t.value)
    near = [t for t in done if t.value <= best.value * 1.005 and t.number != best.number]
    compact = min(near or done, key=lambda t: t.user_attrs["n_params"])
    deep = lambda t: len(t.user_attrs["hidden"]) >= 4
    shallow = lambda t: len(t.user_attrs["hidden"]) <= 2
    other_side = shallow if deep(best) else deep
    pool = [t for t in done if other_side(t) and t.number not in (best.number, compact.number)]
    pool = pool or [t for t in done if t.number not in (best.number, compact.number)]
    contrast = min(pool, key=lambda t: t.value)
    return {"A_melhor": best, "B_compacta": compact, "C_contraste": contrast}


def main():
    prep = load_prep()
    device = get_device()
    train, val = load_encoded("train"), load_encoded("val")
    tests = {"test22": load_encoded("test22"), "test23": load_encoded("test23")}

    study = optuna.load_study(study_name="enem_mlp", storage=STORAGE)
    chosen = select_trials(study)
    configs = {"linear": TrainConfig(hidden=[], emb_dim=16, lr=3e-3, batch_size=4096)}
    for name, t in chosen.items():
        configs[name] = trial_to_config(t)
    for cfg in configs.values():
        cfg.max_epochs, cfg.patience, cfg.scheduler = 40, 5, "plateau"
    meta = {name: {"trial": t.number, "val_rmse_optuna": t.value, "params": t.params, "hidden": t.user_attrs["hidden"]}
            for name, t in chosen.items()}
    (C.TAB_DIR / "selecao_arquiteturas.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, indent=1, ensure_ascii=False), flush=True)

    rows = []
    # referência: preditor constante (média do treino)
    for split, enc in {"val": val, **tests}.items():
        y = prep.inverse_y(enc.y)
        m = regression_metrics(y, np.full_like(y, prep.y_mean_))
        rows.append({"modelo": "media_treino", "seed": 0, "split": split, **m, "n_params": 1})

    preds = {}
    for name, cfg in configs.items():
        for seed in SEEDS:
            cfg.seed = seed
            res = fit(cfg, prep, train, val, verbose=True)
            model = res["model"]
            extra = {"modelo": name, "seed": seed}
            for split, enc in tests.items():
                data = DeviceData(enc, device)
                extra[split] = evaluate(model, data, prep)
                rows.append({"modelo": name, "seed": seed, "split": split, **extra[split], "n_params": res["n_params"],
                             "train_time_s": res["train_time_s"], "infer_time_100k_s": res["infer_time_100k_s"],
                             "best_epoch": res["best_epoch"]})
                if seed == 0:
                    preds[(name, split)] = prep.inverse_y(predict(model, data).cpu().numpy())
                del data
            rows.append({"modelo": name, "seed": seed, "split": "val", **res["val"], "n_params": res["n_params"],
                         "train_time_s": res["train_time_s"], "infer_time_100k_s": res["infer_time_100k_s"],
                         "best_epoch": res["best_epoch"]})
            save_result(res, "finais", f"{name}_seed{seed}", extra)
            if seed == 0:
                (C.OUTPUTS / "modelos").mkdir(exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "config": res["config"]},
                           C.OUTPUTS / "modelos" / f"{name}.pt")
            print(f"== {name} seed {seed}: val {res['val']['rmse']:.2f}  test22 {extra['test22']['rmse']:.2f}  "
                  f"test23 {extra['test23']['rmse']:.2f}  ({res['train_time_s']:.0f}s)", flush=True)
            del model, res
            if device.type == "mps":
                torch.mps.empty_cache()

    pd.DataFrame(rows).to_csv(C.TAB_DIR / "resultados_finais_por_seed.csv", index=False)
    for split in tests:
        pd.DataFrame({name: p for (name, s), p in preds.items() if s == split}).to_parquet(
            C.OUTPUTS / f"predicoes_{split}.parquet", index=False)


if __name__ == "__main__":
    main()
