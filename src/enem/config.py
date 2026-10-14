"""Caminhos, colunas e constantes compartilhadas pelo projeto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"
RUNS_DIR = OUTPUTS / "runs"

SEED = 42
YEARS = (2022, 2023)
TRAIN_YEAR = 2022   # treino, validação e teste "no mesmo ano"
TEST_YEAR = 2023    # teste temporal (ano seguinte)

# Notas que compõem o alvo.
GRADE_COLS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
PRESENCE_COLS = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
TARGET = "NOTA_MEDIA"

# Questionário socioeconômico respondido NA INSCRIÇÃO (antes da prova).
Q_ESCOLARIDADE = ["Q001", "Q002"]                      # pai / mãe
Q_RENDA = ["Q006"]
Q_PESSOAS = ["Q005"]                                   # nº de moradores (numérica)
Q_RESIDENCIA = [f"Q{i:03d}" for i in range(7, 24)]     # Q007..Q023: bens e cômodos
Q_DIGITAL = ["Q024", "Q025"]                           # computador / internet

# Entradas categóricas (codificadas one-hot, ajustadas só no treino).
CAT_FEATURES = (
    ["TP_FAIXA_ETARIA", "TP_ST_CONCLUSAO", "TP_ANO_CONCLUIU", "TP_ESCOLA",
     "TP_DEPENDENCIA_ADM_ESC", "TP_LOCALIZACAO_ESC", "SG_UF_PROVA"]
    + Q_ESCOLARIDADE + Q_RENDA + Q_RESIDENCIA + Q_DIGITAL
)
NUM_FEATURES = Q_PESSOAS
# Município de aplicação da prova: ~5,5 mil categorias -> camada de embedding.
MUNI_FEATURE = "CO_MUNICIPIO_PROVA"

# Variáveis lidas apenas para análise de erro por grupo (NÃO entram no modelo).
AUDIT_COLS = ["TP_SEXO", "TP_COR_RACA", "IN_TREINEIRO", "TP_LINGUA", "TP_STATUS_REDACAO"]

USECOLS = sorted(set(
    ["NU_ANO", MUNI_FEATURE, "NO_MUNICIPIO_PROVA"] + CAT_FEATURES + NUM_FEATURES
    + AUDIT_COLS + GRADE_COLS + PRESENCE_COLS
))
