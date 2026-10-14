"""Monta (e guarda em cache) as partições e os dados codificados.

Protocolo:
  ENEM 2022 -> treino (80%) / validação (10%) / teste-2022 (10%), estratificado por decis do alvo
  ENEM 2023 -> teste temporal (todos os participantes filtrados)
O pré-processador é ajustado só no treino de 2022.
"""
import pickle

import numpy as np
import pandas as pd

from . import config as C
from .data import load_year, split_train_year
from .features import Encoded, Preprocessor

SPLITS = ("train", "val", "test22", "test23")
_CACHE = C.PROCESSED_DIR / "cache"


def build_splits() -> dict:
    df22 = load_year(C.TRAIN_YEAR)
    train, val, test22 = split_train_year(df22)
    test23 = load_year(C.TEST_YEAR)
    return {"train": train, "val": val, "test22": test22, "test23": test23}


def build_cache():
    splits = build_splits()
    prep = Preprocessor().fit(splits["train"])
    _CACHE.mkdir(parents=True, exist_ok=True)
    with open(_CACHE / "prep.pkl", "wb") as fh:
        pickle.dump(prep, fh)
    for name, df in splits.items():
        enc = prep.transform(df)
        np.savez(_CACHE / f"{name}.npz", cat=enc.cat, num=enc.num, muni=enc.muni, y=enc.y)
        # colunas originais usadas na análise de erros (alvo, notas, grupos)
        keep = [C.TARGET] + C.GRADE_COLS + C.AUDIT_COLS + C.CAT_FEATURES + C.NUM_FEATURES + [C.MUNI_FEATURE]
        df[keep].to_parquet(_CACHE / f"{name}_meta.parquet", index=False)
    return prep


def load_prep() -> Preprocessor:
    with open(_CACHE / "prep.pkl", "rb") as fh:
        return pickle.load(fh)


def load_encoded(name: str) -> Encoded:
    z = np.load(_CACHE / f"{name}.npz")
    return Encoded(z["cat"], z["num"], z["muni"], z["y"])


def load_meta(name: str) -> pd.DataFrame:
    return pd.read_parquet(_CACHE / f"{name}_meta.parquet")


def subsample(enc: Encoded, n: int, seed: int = C.SEED) -> Encoded:
    if n >= len(enc):
        return enc
    idx = np.sort(np.random.default_rng(seed).choice(len(enc), size=n, replace=False))
    return enc.subset(idx)
