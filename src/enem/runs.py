"""Gravação e leitura dos resultados de cada treino (JSON por execução)."""
import json

import pandas as pd

from . import config as C


def save_result(result: dict, experiment: str, name: str, extra: dict | None = None):
    out = {k: v for k, v in result.items() if k not in ("model", "final_state")}
    out["name"] = name
    out.update(extra or {})
    path = C.RUNS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


def load_results(experiment: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((C.RUNS_DIR / experiment).glob("*.json"))]


def summary_table(results: list[dict], keys: list[str]) -> pd.DataFrame:
    rows = []
    for r in results:
        hist = pd.DataFrame(r["history"])
        best = hist.loc[hist["val_rmse"].idxmin()]
        last = hist.iloc[-1]
        row = {k: r.get(k) for k in keys}
        row.update({
            "n_params": r["n_params"],
            "melhor_epoca": int(best["epoch"]),
            "val_rmse_melhor": best["val_rmse"],
            "treino_rmse_melhor": best["train_rmse"],
            "val_r2_melhor": best["val_r2"],
            "val_rmse_final": last["val_rmse"],
            "treino_rmse_final": last["train_rmse"],
            "gap_final": last["val_rmse"] - last["train_rmse"],
            "tempo_treino_s": r["train_time_s"],
            "tempo_epoca_s": r["time_per_epoch_s"],
        })
        rows.append(row)
    return pd.DataFrame(rows)
