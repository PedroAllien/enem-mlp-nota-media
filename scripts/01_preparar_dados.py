"""Gera data/processed/enem_{ano}.parquet a partir dos zips do INEP."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enem import config as C
from enem.data import build_year

for year in C.YEARS:
    print(json.dumps(build_year(year), ensure_ascii=False), flush=True)
