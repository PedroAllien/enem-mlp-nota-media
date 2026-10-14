"""Pré-processamento ajustado SOMENTE no conjunto de treino.

- Categóricas: cada coluna vira um bloco one-hot. Valor ausente é uma categoria
  própria ("NA"); categoria nunca vista no treino vira um vetor de zeros.
  Guardamos só os índices das posições "1" (int16) e o vetor one-hot é montado
  na GPU dentro do modelo, o que economiza memória (2 M linhas x ~250 colunas).
- Numérica (Q005, nº de moradores): padronização z-score + indicador de ausente.
- Município: índice inteiro para uma camada de embedding. Municípios com menos
  de `min_muni_count` participantes no treino vão para o índice 0 ("raro/novo").
- Alvo: padronizado (média/desvio do treino); as métricas voltam para pontos.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


def _as_key(s: pd.Series) -> pd.Series:
    """Normaliza para string, de modo que 1, 1.0 e "1" sejam a mesma categoria."""
    out = s.astype("string")
    out = out.str.replace(r"\.0$", "", regex=True)
    return out.fillna("NA")


@dataclass
class Encoded:
    cat: np.ndarray      # (N, n_cat_cols) int16: posição do "1" no vetor one-hot, -1 = desconhecida
    num: np.ndarray      # (N, n_num) float32
    muni: np.ndarray     # (N,) int32
    y: np.ndarray        # (N,) float32 padronizado

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        return Encoded(self.cat[idx], self.num[idx], self.muni[idx], self.y[idx])


@dataclass
class Preprocessor:
    cat_cols: list = field(default_factory=lambda: list(C.CAT_FEATURES))
    num_cols: list = field(default_factory=lambda: list(C.NUM_FEATURES))
    muni_col: str = C.MUNI_FEATURE
    min_muni_count: int = 30

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.categories_ = {}
        self.offsets_ = {}
        offset = 0
        for col in self.cat_cols:
            cats = sorted(_as_key(df[col]).unique().tolist())
            self.categories_[col] = {c: i for i, c in enumerate(cats)}
            self.offsets_[col] = offset
            offset += len(cats)
        self.n_onehot_ = offset

        self.num_stats_ = {}
        for col in self.num_cols:
            x = pd.to_numeric(df[col], errors="coerce")
            self.num_stats_[col] = (float(x.median()), float(x.mean()), float(x.std()))
        self.n_num_ = 2 * len(self.num_cols)   # valor padronizado + indicador de ausente

        counts = df[self.muni_col].value_counts()
        keep = sorted(counts[counts >= self.min_muni_count].index.tolist())
        self.muni_index_ = {int(m): i + 1 for i, m in enumerate(keep)}
        self.n_muni_ = len(keep) + 1

        self.y_mean_ = float(df[C.TARGET].mean())
        self.y_std_ = float(df[C.TARGET].std())
        return self

    def transform(self, df: pd.DataFrame) -> Encoded:
        n = len(df)
        cat = np.empty((n, len(self.cat_cols)), dtype=np.int16)
        for j, col in enumerate(self.cat_cols):
            local = _as_key(df[col]).map(self.categories_[col])
            pos = local + self.offsets_[col]
            cat[:, j] = pos.fillna(-1).to_numpy(dtype=np.int16)

        num = np.empty((n, self.n_num_), dtype=np.float32)
        for j, col in enumerate(self.num_cols):
            med, mu, sd = self.num_stats_[col]
            x = pd.to_numeric(df[col], errors="coerce")
            num[:, 2 * j] = ((x.fillna(med) - mu) / sd).to_numpy(dtype=np.float32)
            num[:, 2 * j + 1] = x.isna().to_numpy(dtype=np.float32)

        muni = df[self.muni_col].map(self.muni_index_).fillna(0).to_numpy(dtype=np.int32)
        y = ((df[C.TARGET] - self.y_mean_) / self.y_std_).to_numpy(dtype=np.float32)
        return Encoded(cat, num, muni, y)

    def inverse_y(self, y_std: np.ndarray) -> np.ndarray:
        return y_std * self.y_std_ + self.y_mean_

    def feature_names(self) -> list:
        names = []
        for col in self.cat_cols:
            names += [f"{col}={c}" for c in self.categories_[col]]
        for col in self.num_cols:
            names += [f"{col}_z", f"{col}_ausente"]
        return names
