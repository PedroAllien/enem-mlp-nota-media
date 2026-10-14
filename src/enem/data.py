"""Leitura dos microdados do INEP, filtragem e construção do alvo.

O CSV (≈1,8 GB por ano) é lido direto de dentro do zip, em blocos, só com as
colunas necessárias. O resultado filtrado é salvo em parquet.
"""
import json
import zipfile

import numpy as np
import pandas as pd

from . import config as C


def _csv_member(zf: zipfile.ZipFile, year: int) -> str:
    name = f"MICRODADOS_ENEM_{year}.csv"
    for member in zf.namelist():
        if member.upper().endswith(name.upper()):
            return member
    raise FileNotFoundError(f"{name} não encontrado no zip")


def build_year(year: int, chunksize: int = 500_000) -> dict:
    """Lê o ano, aplica os filtros e grava data/processed/enem_{year}.parquet.

    Devolve o "funil" de registros para documentar no relatório.
    """
    zpath = C.RAW_DIR / f"microdados_enem_{year}.zip"
    funnel = {"ano": year, "inscritos": 0, "presentes_4_provas": 0,
              "notas_nao_nulas": 0, "sem_nota_objetiva_zero": 0}
    parts = []
    with zipfile.ZipFile(zpath) as zf, zf.open(_csv_member(zf, year)) as fh:
        reader = pd.read_csv(fh, sep=";", encoding="latin-1", usecols=C.USECOLS,
                             chunksize=chunksize, low_memory=False)
        for chunk in reader:
            funnel["inscritos"] += len(chunk)
            # 1) presente nas quatro provas objetivas (exclui faltantes e eliminados)
            chunk = chunk[(chunk[C.PRESENCE_COLS] == 1).all(axis=1)]
            funnel["presentes_4_provas"] += len(chunk)
            # 2) as cinco notas precisam existir
            chunk = chunk.dropna(subset=C.GRADE_COLS)
            funnel["notas_nao_nulas"] += len(chunk)
            # 3) nota 0 em prova objetiva com presença = registro anômalo (TRI não
            #    produz 0 para quem respondeu). Redação 0 é mantida: é um resultado
            #    legítimo (fuga ao tema, texto insuficiente etc.).
            objetivas = C.GRADE_COLS[:4]
            chunk = chunk[(chunk[objetivas] > 0).all(axis=1)]
            funnel["sem_nota_objetiva_zero"] += len(chunk)
            parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    # TP_ANO_CONCLUIU conta "anos antes da prova" (1 = ano anterior), mas o último
    # código é "antes de 2007": 16 em 2022 e 17 em 2023. Agrupamos em 16 = "16 anos
    # ou mais" para que o código signifique o mesmo nos dois anos.
    df["TP_ANO_CONCLUIU"] = df["TP_ANO_CONCLUIU"].clip(upper=16)
    df[C.TARGET] = df[C.GRADE_COLS].mean(axis=1).astype("float32")
    for col in C.GRADE_COLS:
        df[col] = df[col].astype("float32")
    df = df.drop(columns=C.PRESENCE_COLS)

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.PROCESSED_DIR / f"enem_{year}.parquet", index=False)
    (C.PROCESSED_DIR / f"funil_{year}.json").write_text(json.dumps(funnel, indent=2))
    return funnel


def load_year(year: int) -> pd.DataFrame:
    return pd.read_parquet(C.PROCESSED_DIR / f"enem_{year}.parquet")


def split_train_year(df: pd.DataFrame, val_frac=0.10, test_frac=0.10, seed=C.SEED):
    """Divide o ano de treino em treino/validação/teste, estratificando por
    decis do alvo para manter a mesma distribuição de notas nas três partes."""
    from sklearn.model_selection import train_test_split

    strata = pd.qcut(df[C.TARGET], 10, labels=False, duplicates="drop")
    idx = np.arange(len(df))
    idx_trval, idx_test = train_test_split(idx, test_size=test_frac,
                                           stratify=strata, random_state=seed)
    idx_train, idx_val = train_test_split(idx_trval, test_size=val_frac / (1 - test_frac),
                                          stratify=strata.iloc[idx_trval], random_state=seed)
    return (df.iloc[idx_train].reset_index(drop=True),
            df.iloc[idx_val].reset_index(drop=True),
            df.iloc[idx_test].reset_index(drop=True))
