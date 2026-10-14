"""Gera os notebooks a partir das células abaixo (fonte única, fácil de revisar).

Uso: python scripts/build_notebooks.py && jupyter nbconvert --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


SETUP = code("""
import sys, json
sys.path.insert(0, "../src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from enem import config as C, plots
from enem.plots import SERIES, INK_2, MUTED
plots.setup()
pd.set_option("display.max_columns", 50, "display.width", 160, "display.float_format", "{:,.2f}".format)
""")

EDA = [
    md("""
# 01 · Análise exploratória — ENEM 2022 e 2023

**Problema.** Prever a nota média de um participante do ENEM (média de Ciências da Natureza,
Ciências Humanas, Linguagens, Matemática e Redação) usando só informações conhecidas **antes**
da prova: o questionário socioeconômico respondido na inscrição, a situação escolar, a faixa
etária e o local da prova. É um problema de **regressão**; a unidade de observação é o participante.

**Fonte.** Microdados do ENEM, INEP — <https://dados.gov.br/dados/conjuntos-dados/inep-microdados-do-enem>.
2022 é usado para treino/validação/teste e 2023 como teste temporal ("o modelo treinado num ano
funciona no ano seguinte?"). 2024 e 2025 não servem: o INEP separou questionário e notas em
arquivos sem chave em comum (Leia-Me ENEM 2024, seção 3).
"""),
    SETUP,
    code("""
from enem.data import load_year
df = {2022: load_year(2022), 2023: load_year(2023)}
{y: d.shape for y, d in df.items()}
"""),
    md("## 1. Filtragem dos registros\n\nMantemos quem esteve presente nas quatro provas objetivas e tem as cinco notas. "
       "Registros com nota 0 numa prova objetiva, mesmo com presença, são tratados como anômalos e removidos "
       "(a TRI não gera 0 para quem respondeu). Redação 0 é mantida: é um resultado previsto nas regras "
       "(fuga ao tema, texto insuficiente, em branco etc.)."),
    code("""
funil = pd.DataFrame([json.loads((C.PROCESSED_DIR / f"funil_{y}.json").read_text()) for y in C.YEARS]).set_index("ano").T
funil.index = ["Inscritos", "Presentes nas 4 provas objetivas", "Com as 5 notas", "Sem nota objetiva igual a 0"]
funil["% 2022"] = 100 * funil[2022] / funil.loc["Inscritos", 2022]
funil["% 2023"] = 100 * funil[2023] / funil.loc["Inscritos", 2023]
funil.to_csv(C.TAB_DIR / "funil.csv"); funil
"""),
    md("## 2. Variáveis\n\nTodas as entradas são categóricas, exceto o número de moradores (Q005). "
       "O município de residência foi retirado dos microdados pelo INEP (LGPD); usamos o **município de aplicação "
       "da prova**, que normalmente é o mesmo ou um vizinho."),
    code("""
DESCR = {
 "TP_FAIXA_ETARIA": ("Faixa etária", "ordinal (20 faixas)"),
 "TP_ST_CONCLUSAO": ("Situação de conclusão do EM", "nominal"),
 "TP_ANO_CONCLUIU": ("Anos desde a conclusão do EM (0 = não informado; 16 = 16 ou mais)", "ordinal"),
 "TP_ESCOLA": ("Tipo de escola do EM (1 = não respondeu, 2 = pública, 3 = privada)", "nominal"),
 "TP_DEPENDENCIA_ADM_ESC": ("Dependência administrativa da escola", "nominal"),
 "TP_LOCALIZACAO_ESC": ("Localização da escola (urbana/rural)", "nominal"),
 "SG_UF_PROVA": ("UF da prova", "nominal (27)"),
 "CO_MUNICIPIO_PROVA": ("Município da prova", "nominal (alta cardinalidade)"),
 "Q001": ("Escolaridade do pai/responsável", "ordinal + 'não sei'"),
 "Q002": ("Escolaridade da mãe/responsável", "ordinal + 'não sei'"),
 "Q005": ("Nº de moradores na residência", "numérica (1–20)"),
 "Q006": ("Renda familiar mensal (17 faixas)", "ordinal"),
 **{f"Q{i:03d}": ("Característica/bem da residência", "ordinal") for i in range(7, 24)},
 "Q024": ("Computador na residência", "ordinal"),
 "Q025": ("Acesso à internet", "binária"),
}
cols = C.CAT_FEATURES + C.NUM_FEATURES + [C.MUNI_FEATURE]
var = pd.DataFrame({
    "descrição": [DESCR[c][0] for c in cols],
    "tipo": [DESCR[c][1] for c in cols],
    "n_categorias": [df[2022][c].nunique() for c in cols],
    "% ausente 2022": [100 * df[2022][c].isna().mean() for c in cols],
    "% ausente 2023": [100 * df[2023][c].isna().mean() for c in cols],
}, index=cols)
var.to_csv(C.TAB_DIR / "variaveis.csv"); var
"""),
    md("""
### 2.1 Valores ausentes e respostas "não informativas"

O questionário não tem valores ausentes entre os presentes. A ausência relevante é **estrutural**:
os campos da escola (`TP_DEPENDENCIA_ADM_ESC`, `TP_LOCALIZACAO_ESC`) só são preenchidos para quem
conclui o ensino médio no ano da prova. Além disso, há categorias que funcionam como ausência
("não respondeu" em `TP_ESCOLA`, "não sei" em Q001/Q002). Em vez de imputar, mantemos a ausência como uma
**categoria própria** no one-hot: o fato de não saber ou não responder também carrega informação.
"""),
    code("""
nao_info = pd.DataFrame({
  y: {"TP_ESCOLA = não respondeu": 100 * (d["TP_ESCOLA"] == 1).mean(),
      "Escola (dependência adm.) ausente": 100 * d["TP_DEPENDENCIA_ADM_ESC"].isna().mean(),
      "Q001 = não sei (pai)": 100 * (d["Q001"] == "H").mean(),
      "Q002 = não sei (mãe)": 100 * (d["Q002"] == "H").mean(),
      "TP_ANO_CONCLUIU = não informado": 100 * (d["TP_ANO_CONCLUIU"] == 0).mean()}
  for y, d in df.items()})
nao_info.round(1)
"""),
    code("""
d = df[2022]
print("TP_ESCOLA por situação de conclusão (2022, % na linha):")
pd.crosstab(d["TP_ST_CONCLUSAO"], d["TP_ESCOLA"], normalize="index").mul(100).round(1)
"""),
    md("### 2.2 Duplicados\n\nCada linha é uma inscrição. Verificamos (a) linhas idênticas em entradas **e** notas, que "
       "indicariam registro duplicado, e (b) participantes com o mesmo perfil de entrada. Pessoas com o mesmo perfil e notas "
       "diferentes mostram o erro irredutível: nenhum modelo acerta duas notas diferentes para entradas idênticas."),
    code("""
feat = C.CAT_FEATURES + C.NUM_FEATURES + [C.MUNI_FEATURE]
d = df[2022]
dup_full = d.duplicated(subset=feat + C.GRADE_COLS).sum()
dup_x = d.duplicated(subset=feat).sum()
print(f"linhas duplicadas (entradas + 5 notas): {dup_full:,}")
print(f"linhas com perfil de entrada repetido:  {dup_x:,} ({100*dup_x/len(d):.1f}%)")
# ruído irredutível estimado: variância do alvo DENTRO de perfis idênticos (perfis com >= 5 pessoas)
g = d.groupby(feat, observed=True)[C.TARGET]
stats_g = g.agg(["count", "var"])
big = stats_g[stats_g["count"] >= 5]
print(f"perfis com >=5 participantes: {len(big):,} grupos, {big['count'].sum():,} pessoas")
print(f"desvio-padrão médio da nota dentro do mesmo perfil: {np.sqrt(np.average(big['var'], weights=big['count'])):.1f} pontos")
"""),
    md("## 3. Variável alvo: nota média\n\n`NOTA_MEDIA` = média simples das 5 notas (escala ENEM, em pontos). "
       "As quatro provas objetivas usam a TRI (escala com média 500 e desvio 100 na referência de 2009); a redação vai de 0 a 1000."),
    code("""
from scipy import stats
def resumo(s):
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    return pd.Series({"n": len(s), "média": s.mean(), "desvio": s.std(), "mín": s.min(), "p1": s.quantile(.01),
        "p25": q1, "mediana": s.median(), "p75": q3, "p99": s.quantile(.99), "máx": s.max(),
        "assimetria": stats.skew(s), "curtose (excesso)": stats.kurtosis(s),
        "% fora 1,5·IQR": 100 * ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean()})
alvo = pd.DataFrame({y: resumo(d[C.TARGET]) for y, d in df.items()})
alvo.to_csv(C.TAB_DIR / "alvo_resumo.csv"); alvo
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.arange(250, 900, 5)
for i, (y, d) in enumerate(df.items()):
    ax.hist(d[C.TARGET], bins=bins, histtype="step", linewidth=2, color=SERIES[i], density=True, label=str(y))
    ax.axvline(d[C.TARGET].mean(), color=SERIES[i], linewidth=1)
ax.set_xlabel("Nota média (pontos)"); ax.set_ylabel("Densidade")
ax.set_title("Distribuição da nota média: unimodal e quase simétrica; 2023 um pouco mais dispersa")
ax.legend(); plots.save(fig, "eda_alvo_hist")
"""),
    code("""
nomes = {"NU_NOTA_CN": "Ciências da Natureza", "NU_NOTA_CH": "Ciências Humanas", "NU_NOTA_LC": "Linguagens",
         "NU_NOTA_MT": "Matemática", "NU_NOTA_REDACAO": "Redação"}
fig, axes = plt.subplots(1, 5, figsize=(13, 2.8), sharey=False)
for ax, c in zip(axes, C.GRADE_COLS):
    ax.hist(df[2022][c], bins=60, color=SERIES[0], alpha=.85)
    ax.set_title(nomes[c], fontsize=9); ax.set_yticks([])
fig.suptitle("Notas por prova (2022) — a redação é discreta (múltiplos de 20) e tem massa em 0", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); plots.save(fig, "eda_notas_por_prova")
"""),
    code("""
corr = df[2022][C.GRADE_COLS].corr().rename(index=nomes, columns=nomes)
print("Correlação entre as cinco notas (2022) — todas positivas e moderadas a altas, o que justifica um alvo agregado:")
corr.round(2)
"""),
    code("""
red0 = {y: 100 * (d["NU_NOTA_REDACAO"] == 0).mean() for y, d in df.items()}
print("% com redação = 0:", {k: round(v, 2) for k, v in red0.items()})
df[2022].loc[df[2022]["NU_NOTA_REDACAO"] == 0, "TP_STATUS_REDACAO"].value_counts()
"""),
    md("## 4. Relação entre entradas e alvo\n\nMedimos a força de cada variável sozinha pela **razão de correlação η²** "
       "(fração da variância do alvo explicada pelas médias por categoria). É uma medida univariada: não considera redundância entre variáveis."),
    code("""
def eta2(x, y):
    g = pd.DataFrame({"x": x.astype("string").fillna("NA"), "y": y}).groupby("x")["y"]
    return float((g.count() * (g.mean() - y.mean())**2).sum() / ((y - y.mean())**2).sum())
d = df[2022]
ROTULO = {"Q001": "Escolaridade do pai", "Q002": "Escolaridade da mãe", "Q005": "Nº de moradores", "Q006": "Renda familiar",
  "Q007": "Empregado doméstico", "Q008": "Banheiros", "Q009": "Quartos", "Q010": "Carros", "Q011": "Motocicletas",
  "Q012": "Geladeira", "Q013": "Freezer", "Q014": "Máq. lavar roupa", "Q015": "Máq. secar roupa", "Q016": "Micro-ondas",
  "Q017": "Lava-louça", "Q018": "Aspirador", "Q019": "TVs", "Q020": "DVD", "Q021": "TV por assinatura", "Q022": "Celulares",
  "Q023": "Telefone fixo", "Q024": "Computadores", "Q025": "Internet", "TP_FAIXA_ETARIA": "Faixa etária",
  "TP_ST_CONCLUSAO": "Situação de conclusão", "TP_ANO_CONCLUIU": "Anos desde conclusão", "TP_ESCOLA": "Tipo de escola",
  "TP_DEPENDENCIA_ADM_ESC": "Escola: dependência adm.", "TP_LOCALIZACAO_ESC": "Escola: localização",
  "SG_UF_PROVA": "UF da prova", "CO_MUNICIPIO_PROVA": "Município da prova"}
rank = pd.Series({c: eta2(d[c], d[C.TARGET]) for c in feat}).sort_values()
fig, ax = plt.subplots(figsize=(7, 7.5))
ax.barh([f"{ROTULO[c]} ({c})" for c in rank.index], rank.values, height=0.6, color=SERIES[0])
for i, v in enumerate(rank.values):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
ax.set_xlabel("η² (fração da variância da nota média explicada pela variável isolada)")
ax.set_title("Renda, escolaridade da mãe e bens da casa explicam mais; município vem logo atrás")
ax.grid(axis="y", visible=False); plots.save(fig, "eda_eta2")
rank.sort_values(ascending=False).to_csv(C.TAB_DIR / "eta2.csv")
"""),
    code("""
def media_por(col, ordem=None, rotulos=None):
    g = df[2022].groupby(col, observed=True)[C.TARGET].agg(["mean", "count"])
    if ordem is not None: g = g.reindex(ordem)
    if rotulos is not None: g.index = rotulos
    return g

paineis = [  # (coluna, título, ordem, rótulos, ordinal?)
    ("Q006", "Renda familiar (A = nenhuma … Q = >20 SM)", None, None, True),
    ("Q002", "Escolaridade da mãe (A … G = pós; H = não sei)", None, None, True),
    ("TP_ESCOLA", "Tipo de escola", [1, 2, 3], ["Não resp.", "Pública", "Privada"], False),
    ("Q024", "Computadores em casa (A = 0 … E = 4+)", None, None, True),
    ("Q025", "Internet em casa", ["A", "B"], ["Não", "Sim"], False),
    ("TP_FAIXA_ETARIA", "Faixa etária (1 = <17 … 10 = 25 anos; 11+ = faixas de 5 anos)", list(range(1, 21)), None, True),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, (col, titulo, ordem, rot, ordinal) in zip(axes.flat, paineis):
    g = media_por(col, ordem, rot)
    x = np.arange(len(g))
    # "não sei" (H) não é um nível da escala: fica como ponto solto
    n_linha = len(g) - 1 if col == "Q002" else len(g)
    if ordinal:
        ax.plot(x[:n_linha], g["mean"].values[:n_linha], color=SERIES[0], linewidth=2)
    ax.scatter(x, g["mean"], s=36, color=SERIES[0], edgecolor=plots.SURFACE, linewidth=1.5, zorder=3)
    ax.set_xticks(x, [str(i) for i in g.index], fontsize=8)
    ax.set_title(titulo, fontsize=9); ax.set_ylim(430, 680)
    if not ordinal: ax.set_xlim(-0.6, len(g) - 0.4)
    ax.axhline(df[2022][C.TARGET].mean(), color=MUTED, linewidth=1)
axes[0, 0].set_ylabel("Nota média (pontos)"); axes[1, 0].set_ylabel("Nota média (pontos)")
fig.suptitle("Nota média por categoria (2022); linha cinza = média geral", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); plots.save(fig, "eda_media_por_categoria")
"""),
    code("""
uf = df[2022].groupby("SG_UF_PROVA")[C.TARGET].agg(["mean", "count"]).sort_values("mean")
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.bar(uf.index, uf["mean"] - 400, bottom=400, width=0.6, color=SERIES[0])
ax.axhline(df[2022][C.TARGET].mean(), color=MUTED, linewidth=1)
ax.set_ylabel("Nota média (pontos)"); ax.set_ylim(400, 600)
ax.set_title("Nota média por UF da prova (2022): diferença de ~70 pontos entre extremos")
plots.save(fig, "eda_media_por_uf")
"""),
    md("### 4.1 Município\n\nO ENEM é aplicado em ~1,75 mil municípios. A camada de **embedding** aprende um vetor por município "
       "(municípios com menos de 30 participantes no treino caem numa categoria única \"raro/novo\"). "
       "A variação entre municípios é grande mesmo dentro da mesma UF, o que justifica usar o município além da UF."),
    code("""
m = df[2022].groupby(C.MUNI_FEATURE).agg(n=(C.TARGET, "size"), media=(C.TARGET, "mean"), uf=("SG_UF_PROVA", "first"))
print(f"municípios: {len(m):,}; com >=30 participantes: {(m['n']>=30).sum():,} ({100*m.loc[m['n']>=30,'n'].sum()/m['n'].sum():.2f}% dos participantes)")
print("participantes por município (quantis):", m["n"].quantile([.1,.25,.5,.75,.9,.99]).round(0).to_dict())
dentro_uf = m[m["n"] >= 30].groupby("uf")["media"].agg(lambda s: s.max() - s.min())
print(f"amplitude média da nota entre municípios de uma mesma UF: {dentro_uf.mean():.0f} pontos")
"""),
    md("## 5. Mudança de distribuição entre 2022 e 2023\n\nO teste em 2023 avalia se o modelo continua válido no ano seguinte. "
       "As notas da TRI são equalizadas entre anos, mas a redação e a composição dos participantes mudam. "
       "Comparamos as médias e a **distância de variação total** (TVD, 0 = distribuições iguais, 1 = disjuntas) de cada entrada."),
    code("""
comp = pd.DataFrame({y: d[C.GRADE_COLS + [C.TARGET]].mean() for y, d in df.items()})
comp["diferença"] = comp[2023] - comp[2022]; comp
"""),
    code("""
def tvd(a, b):
    pa = a.astype("string").fillna("NA").value_counts(normalize=True)
    pb = b.astype("string").fillna("NA").value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()
shift = pd.Series({c: tvd(df[2022][c], df[2023][c]) for c in feat}).sort_values(ascending=False)
shift.head(10).round(3)
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 3.4))
letras = list("ABCDEFGHIJKLMNOPQ")
for i, (y, d) in enumerate(df.items()):
    p = d["Q006"].value_counts(normalize=True).reindex(letras).fillna(0) * 100
    ax.plot(range(len(letras)), p.values, marker="o", color=SERIES[i], label=str(y))
ax.set_xticks(range(len(letras)), letras); ax.set_ylabel("% dos participantes")
ax.set_xlabel("Faixa de renda (Q006), em múltiplos do salário mínimo de cada ano")
ax.set_title("Distribuição de renda declarada é estável entre os anos"); ax.legend()
plots.save(fig, "eda_renda_2022_2023")
"""),
    md("""
## 6. Risco de vazamento e variáveis excluídas

| Variável | Motivo da exclusão |
|---|---|
| `NU_NOTA_*`, `NU_NOTA_COMP*` | compõem o alvo |
| `TX_RESPOSTAS_*`, `TX_GABARITO_*` | respostas e gabaritos da própria prova |
| `TP_PRESENCA_*`, `TP_STATUS_REDACAO` | só existem depois da prova (e definem o filtro) |
| `CO_PROVA_*` | cor do caderno, definida no dia da prova |
| `TP_SEXO`, `TP_COR_RACA` | conhecidas antes da prova, mas deixadas de fora de propósito para não usar atributos protegidos como preditores; são usadas só na **análise de erro por grupo** |
| `Q003`, `Q004`, `TP_LINGUA`, `TP_ESTADO_CIVIL` | conhecidas antes da prova, mas fora do escopo definido (ver relatório) |

Todas as entradas usadas vêm da ficha de inscrição, preenchida meses antes da prova.

## 7. Conclusões da análise exploratória

- ~2,34 milhões de participantes válidos em 2022 e ~2,66 milhões em 2023; não há ausências no questionário,
  só ausência estrutural nos campos da escola, tratada como categoria própria.
- O alvo é unimodal e quase simétrico (assimetria ≈ 0,05, curtose ≈ 0), e só ~0,5% das notas fica fora de 1,5·IQR. Esses
  valores extremos são notas reais (redação 0, notas máximas), não erros de registro, e são mantidos.
- 4,2% dos participantes de 2022 tiraram 0 na redação (em branco, fuga ao tema, cópia do texto motivador). Isso puxa a nota
  média para baixo de um jeito que o perfil socioeconômico dificilmente prevê.
- Renda, escolaridade dos pais, bens da residência e local da prova são os sinais mais fortes; cada um sozinho explica até ~20% da variância.
- Só 0,7% dos participantes repetem exatamente o perfil de entrada (o questionário tem muitas combinações), mas, nos perfis
  repetidos, a nota varia ~73 pontos (desvio-padrão) entre pessoas idênticas para o modelo. O **erro irredutível é grande**,
  então não se deve esperar R² perto de 1 com informação socioeconômica.
"""),
]


RES = [
    md("""
# 02 · Experimentos e resultados

Este notebook só **lê** os registros gerados pelos scripts em `scripts/` (cada treino salva um JSON em
`outputs/runs/`, e o estudo Optuna fica em `outputs/optuna.db`) e produz as tabelas e figuras do relatório.

| Script | Experimento |
|---|---|
| `02_capacidade.py` | profundidade × largura, em duas escalas de treino (20 mil e 500 mil exemplos) |
| `03_regularizacao.py` | dropout, L1, L2, BatchNorm, LayerNorm e early stopping numa MLP 4×512 |
| `04_optuna.py` | busca de hiperparâmetros (TPE + poda por mediana), 60 trials |
| `05_modelos_finais.py` | 3 arquiteturas selecionadas + referências, treino completo, 3 sementes, testes 2022 e 2023 |
"""),
    SETUP,
    code("""
from enem.runs import load_results, summary_table
from enem.pipeline import load_meta
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
ESCALAS = {"n20k": "20 mil exemplos de treino", "n500k": "500 mil exemplos de treino"}
def curvas(ax, r, titulo=None, ymax=None):
    h = pd.DataFrame(r["history"])
    ax.plot(h["epoch"], h["train_rmse"], color=SERIES[0], label="treino")
    ax.plot(h["epoch"], h["val_rmse"], color=SERIES[1], label="validação")
    b = h.loc[h["val_rmse"].idxmin()]
    ax.scatter([b["epoch"]], [b["val_rmse"]], s=36, color=SERIES[1], edgecolor=plots.SURFACE, linewidth=1.5, zorder=3)
    if titulo: ax.set_title(titulo, fontsize=9)
    if ymax: ax.set_ylim(0, ymax)
"""),
    md("## 1. Capacidade: profundidade × largura\n\nSem regularização, LR constante e todas as épocas (120 na escala de 20 mil, 30 na de 500 mil). "
       "`melhor` = melhor época na validação (o que o early stopping pegaria); `final` = última época. Validação: 100 mil exemplos de 2022."),
    code("""
cap = load_results("capacidade")
tcap = summary_table(cap, ["escala", "profundidade", "largura"]).sort_values(["escala", "profundidade", "largura"])
tcap.to_csv(C.TAB_DIR / "capacidade.csv", index=False)
tcap
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, (esc, titulo) in zip(axes, ESCALAS.items()):
    t = tcap[tcap["escala"] == esc]
    lin = t[t["profundidade"] == 0]["val_rmse_melhor"].iloc[0]
    for i, w in enumerate([32, 128, 512]):
        tw = t[t["largura"] == w].sort_values("profundidade")
        x = [0] + tw["profundidade"].tolist(); y = [lin] + tw["val_rmse_melhor"].tolist()
        ax.plot(x, y, marker="o", color=SERIES[i], label=f"largura {w}")
        yf = [t[t["profundidade"] == 0]["val_rmse_final"].iloc[0]] + tw["val_rmse_final"].tolist()
    ax.set_title(titulo, fontsize=10); ax.set_xlabel("Camadas ocultas (0 = regressão linear)")
    ax.set_xticks([0, 1, 2, 3, 4, 6])
axes[0].set_ylabel("RMSE de validação na melhor época (pontos)"); axes[0].legend()
fig.suptitle("Melhor RMSE de validação por arquitetura", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); plots.save(fig, "cap_melhor_val")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (esc, titulo) in zip(axes, ESCALAS.items()):
    t = tcap[(tcap["escala"] == esc) & (tcap["profundidade"] > 0)]
    for i, w in enumerate([32, 128, 512]):
        tw = t[t["largura"] == w].sort_values("profundidade")
        ax.plot(tw["profundidade"], tw["gap_final"], marker="o", color=SERIES[i], label=f"largura {w}")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_title(titulo, fontsize=10); ax.set_xlabel("Camadas ocultas"); ax.set_xticks([1, 2, 3, 4, 6])
axes[0].set_ylabel("RMSE validação − RMSE treino\\nna última época (pontos)"); axes[0].legend()
fig.suptitle("Overfitting ao fim do treino (sem regularização)", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); plots.save(fig, "cap_gap")
"""),
    code("""
sel = [("d0_w0", "linear"), ("d1_w32", "1×32"), ("d3_w128", "3×128"), ("d6_w512", "6×512")]
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey="row")
por_nome = {r["name"]: r for r in cap}
for row, esc in enumerate(ESCALAS):
    for col, (k, rot) in enumerate(sel):
        curvas(axes[row, col], por_nome[f"{esc}_{k}"], f"{rot} · {ESCALAS[esc]}")
    axes[row, 0].set_ylabel("RMSE (pontos)")
for ax in axes[1]: ax.set_xlabel("Época")
axes[0, 0].legend()
fig.suptitle("Curvas de treino e validação (ponto = melhor época da validação)", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); plots.save(fig, "cap_curvas")
"""),
    code("""
t = tcap[tcap["escala"] == "n500k"]
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, w in enumerate([32, 128, 512]):
    tw = t[t["largura"] == w]
    ax.scatter(tw["n_params"], tw["tempo_epoca_s"], s=40, color=SERIES[i], label=f"largura {w}", edgecolor=plots.SURFACE, linewidth=1.5)
l = t[t["profundidade"] == 0]
ax.scatter(l["n_params"], l["tempo_epoca_s"], s=40, color=MUTED, label="linear")
ax.set_xscale("log"); ax.set_xlabel("Parâmetros treináveis"); ax.set_ylabel("Segundos por época (500 mil ex., GPU M2)")
ax.set_title("Custo por época cresce com os parâmetros"); ax.legend()
plots.save(fig, "cap_custo")
"""),
    md("## 2. Regularização (MLP 4×512)\n\nCada variante muda **uma** coisa em relação a \"sem regularização\". "
       "`val_restaurado` é o RMSE do modelo devolvido: com early stopping, os pesos da melhor época; sem early stopping, os da última."),
    code("""
reg = load_results("regularizacao")
treg = summary_table(reg, ["escala", "variante"])
treg["val_restaurado"] = [r["val"]["rmse"] for r in reg]
ordem = ["sem_regularizacao", "early_stopping", "dropout_0.2", "dropout_0.5", "l2_1e-4", "l2_1e-3",
         "l1_1e-5", "l1_1e-4", "batchnorm", "layernorm", "dropout_l2_es"]
ROT_REG = {"sem_regularizacao": "sem regularização", "early_stopping": "early stopping (paciência 10)", "dropout_0.2": "dropout 0,2",
           "dropout_0.5": "dropout 0,5", "l2_1e-4": "L2 λ = 1e-4", "l2_1e-3": "L2 λ = 1e-3", "l1_1e-5": "L1 λ = 1e-5", "l1_1e-4": "L1 λ = 1e-4",
           "batchnorm": "BatchNorm", "layernorm": "LayerNorm", "dropout_l2_es": "dropout 0,3 + L2 1e-4 + early stopping"}
treg["ordem"] = treg["variante"].map({v: i for i, v in enumerate(ordem)})
treg = treg.sort_values(["escala", "ordem"]).drop(columns="ordem")
treg.to_csv(C.TAB_DIR / "regularizacao.csv", index=False)
treg[["escala", "variante", "melhor_epoca", "val_rmse_melhor", "treino_rmse_final", "val_rmse_final", "gap_final", "val_restaurado", "tempo_treino_s"]]
"""),
    code("""
por_nome = {r["name"]: r for r in reg}
for esc in ESCALAS:
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True, sharey=True)
    for ax, v in zip(axes.flat, ordem):
        curvas(ax, por_nome[f"{esc}_{v}"], ROT_REG[v])
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]: ax.set_ylabel("RMSE (pontos)")
    axes[0, 0].legend()
    fig.suptitle(f"Curvas por estratégia de regularização · {ESCALAS[esc]}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); plots.save(fig, f"reg_curvas_{esc}")
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 4.6))
y = np.arange(len(ordem))
for i, esc in enumerate(ESCALAS):
    t = treg[treg["escala"] == esc].set_index("variante").reindex(ordem)
    ax.scatter(t["val_restaurado"], y + (i - 0.5) * 0.25, s=40, color=SERIES[i], label=ESCALAS[esc], edgecolor=plots.SURFACE, linewidth=1.5, zorder=3)
ax.set_yticks(y, [ROT_REG[v] for v in ordem]); ax.invert_yaxis()
ax.set_xlabel("RMSE de validação do modelo devolvido (pontos)"); ax.legend(loc="lower right")
ax.set_title("Com 500 mil exemplos, L1/L2 fortes, dropout 0,5 e early stopping seguram o overfitting;\\ncom 20 mil, só o early stopping")
plots.save(fig, "reg_resumo")
"""),
    md("### 2.1 Diagnóstico: por que L1/L2 não seguraram o overfitting com 20 mil exemplos?\n\n"
       "Com 20 mil exemplos, as penalidades deixaram a validação final **pior** que sem regularização. Testamos duas hipóteses "
       "(`scripts/03b_diagnostico_l2.py`): (1) o embedding do município, que não é penalizado, compensaria pesos encolhidos → repetir sem embedding; "
       "(2) a penalidade somada à perda perde efeito no Adam → trocar por weight decay desacoplado (AdamW)."),
    code("""
diag = load_results("diagnostico_l2")
linhas = [{"variante": r["variante"], "embedding": r["config"]["emb_dim"], "otimizador": "AdamW" if r["config"].get("weight_decay", 0) > 0 else "Adam",
           "val_melhor": min(h["val_rmse"] for h in r["history"]), "treino_final": r["history"][-1]["train_rmse"],
           "val_final": r["history"][-1]["val_rmse"]} for r in diag]
for v in ["sem_regularizacao", "l2_1e-3", "l1_1e-4"]:
    r = por_nome[f"n20k_{v}"]
    linhas.append({"variante": v, "embedding": 16, "otimizador": "Adam", "val_melhor": min(h["val_rmse"] for h in r["history"]),
                   "treino_final": r["history"][-1]["train_rmse"], "val_final": r["history"][-1]["val_rmse"]})
tdiag = pd.DataFrame(linhas).sort_values(["embedding", "otimizador", "variante"])
tdiag.to_csv(C.TAB_DIR / "diagnostico_l2.csv", index=False); tdiag
"""),
    md("## 3. Busca de hiperparâmetros (Optuna)"),
    code("""
study = optuna.load_study(study_name="enem_mlp", storage=f"sqlite:///{C.OUTPUTS / 'optuna.db'}")
tr = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
tr["duracao_s"] = tr["duration"].dt.total_seconds()
print(tr["state"].value_counts().to_string())
print(f"tempo total da busca: {tr['duracao_s'].sum()/60:.0f} min; melhor trial: #{study.best_trial.number} com RMSE {study.best_value:.2f}")
comp = tr[tr["state"] == "COMPLETE"].sort_values("value")
cols = ["number", "value", "user_attrs_val_r2", "user_attrs_train_rmse", "user_attrs_hidden", "params_ativacao", "params_dropout",
        "params_normalizacao", "params_l2", "params_lr", "params_batch_size", "params_emb_municipio", "user_attrs_n_params", "duracao_s"]
comp[cols].head(10)
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 3.6))
okm = tr["state"] == "COMPLETE"
ax.scatter(tr.loc[okm, "number"], tr.loc[okm, "value"], s=30, color=SERIES[0], label="completo", edgecolor=plots.SURFACE, linewidth=1.5, zorder=3)
pr = tr[tr["state"] == "PRUNED"]
# trial podado: último valor reportado
ultimo = {t.number: list(t.intermediate_values.values())[-1] for t in study.trials if t.state.name == "PRUNED" and t.intermediate_values}
ax.scatter(list(ultimo), list(ultimo.values()), s=30, color=MUTED, label="podado (último valor)", zorder=2)
melhor = tr.loc[okm].set_index("number")["value"].cummin()
ax.step(melhor.index, melhor.values, where="post", color=SERIES[1], label="melhor até aqui")
ax.set_ylim(tr.loc[okm, "value"].min() - 0.15, tr.loc[okm, "value"].quantile(0.95) + 0.5)
ax.set_xlabel("Trial"); ax.set_ylabel("RMSE de validação (pontos)"); ax.legend()
ax.set_title("Histórico da otimização"); plots.save(fig, "optuna_historico")
"""),
    code("""
imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=C.SEED))
imp = pd.Series(imp).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(imp.index, imp.values, height=0.6, color=SERIES[0])
for i, v in enumerate(imp.values): ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
ax.set_xlabel("Importância (fANOVA): fração da variância do RMSE entre trials"); ax.grid(axis="y", visible=False)
ax.set_title("Quais hiperparâmetros mais mexeram no resultado"); plots.save(fig, "optuna_importancia")
imp.sort_values(ascending=False).round(3)
"""),
    code("""
num = [("params_n_camadas", "Camadas ocultas", False), ("params_largura", "Largura (1ª camada)", True), ("params_lr", "Taxa de aprendizado", True),
       ("params_dropout", "Dropout", False), ("params_l2", "Penalidade L2", True), ("params_batch_size", "Tamanho do lote", False),
       ("params_emb_municipio", "Dimensão do embedding do município", False), ("params_normalizacao", "Normalização", False)]
c = tr[okm]
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
for ax, (col, rot, logx) in zip(axes.flat, num):
    x = c[col].astype(str) if col == "params_batch_size" else c[col]
    if col == "params_batch_size":
        cats = ["512", "1024", "2048", "4096"]; x = x.map({k: i for i, k in enumerate(cats)}); ax.set_xticks(range(len(cats)), cats)
    elif x.dtype == object:
        cats = sorted(x.unique()); x = x.map({k: i for i, k in enumerate(cats)}); ax.set_xticks(range(len(cats)), cats)
    ax.scatter(x, c["value"], s=24, color=SERIES[0], alpha=0.8, edgecolor=plots.SURFACE, linewidth=1)
    if logx: ax.set_xscale("log")
    ax.set_title(rot, fontsize=9)
for ax in axes[:, 0]: ax.set_ylabel("RMSE validação")
axes[0, 0].set_ylim(top=c["value"].quantile(0.95) + 0.5)
fig.suptitle("RMSE de validação por hiperparâmetro (trials completos)", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); plots.save(fig, "optuna_slices")
"""),
    code("""
sel = json.loads((C.TAB_DIR / "selecao_arquiteturas.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, (col, rot) in zip(axes, [("user_attrs_n_params", "Parâmetros treináveis"), ("duracao_s", "Duração do trial (s)")]):
    ax.scatter(c[col], c["value"], s=28, color=MUTED, edgecolor=plots.SURFACE, linewidth=1, label="trials")
    for i, (nome, m) in enumerate(sel.items()):
        r = c[c["number"] == m["trial"]]
        ax.scatter(r[col], r["value"], s=70, color=SERIES[i], edgecolor=plots.SURFACE, linewidth=1.5, zorder=3, label=f"{nome} (#{m['trial']})")
    ax.set_xscale("log"); ax.set_xlabel(rot)
axes[0].set_ylabel("RMSE de validação"); axes[0].set_ylim(top=c["value"].quantile(0.95) + 0.5); axes[0].legend()
fig.suptitle("Custo × desempenho dos trials e arquiteturas selecionadas", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); plots.save(fig, "optuna_pareto")
pd.DataFrame(sel).T
"""),
    code("""
c2 = c.assign(gap=c["value"] - c["user_attrs_train_rmse"])
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, (k, g) in enumerate(c2.groupby("params_normalizacao")):
    ax.scatter(g["params_dropout"], g["gap"], s=30, color=SERIES[i], label=f"normalização: {k}", edgecolor=plots.SURFACE, linewidth=1)
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_xlabel("Dropout"); ax.set_ylabel("RMSE val − RMSE treino (pontos)"); ax.legend()
ax.set_title("Sinal de overfitting nos trials (400 mil exemplos)"); plots.save(fig, "optuna_gap")
"""),
    md("## 4. Avaliação final\n\nArquiteturas retreinadas no **treino completo** (1,87 mi), com early stopping na validação, 3 sementes cada. "
       "**teste-2022** = 10% de 2022 nunca vistos; **teste-2023** = todos os 2,66 mi participantes de 2023 (ano seguinte)."),
    code("""
fr = pd.read_csv(C.TAB_DIR / "resultados_finais_por_seed.csv")
agg = fr.groupby(["modelo", "split"]).agg(rmse=("rmse", "mean"), rmse_dp=("rmse", "std"), mae=("mae", "mean"), r2=("r2", "mean"),
        vies=("bias", "mean"), n_params=("n_params", "first"), treino_s=("train_time_s", "mean"),
        inferencia_100k_s=("infer_time_100k_s", "mean"), melhor_epoca=("best_epoch", "mean")).reset_index()
ordem_m = ["media_treino", "linear", "A_melhor", "B_compacta", "C_contraste"]
agg["o"] = agg["modelo"].map({m: i for i, m in enumerate(ordem_m)})
agg = agg.sort_values(["split", "o"]).drop(columns="o")
agg.to_csv(C.TAB_DIR / "resultados_finais.csv", index=False)
tab = agg.pivot(index="modelo", columns="split", values=["rmse", "mae", "r2"]).reindex(ordem_m)
tab.columns = [f"{m}_{s}" for m, s in tab.columns]
custo = agg[agg["split"] == "test23"].set_index("modelo")[["n_params", "treino_s", "inferencia_100k_s", "melhor_epoca"]]
final = tab[[f"{m}_{s}" for s in ["val", "test22", "test23"] for m in ["rmse", "mae", "r2"]]].join(custo)
final.to_csv(C.TAB_DIR / "tabela_final.csv"); final
"""),
    code("""
print("Desvio-padrão do RMSE entre sementes (pontos):")
agg.pivot(index="modelo", columns="split", values="rmse_dp").reindex(ordem_m).round(3)
"""),
    code("""
fin = {r["name"]: r for r in load_results("finais")}
nomes_m = ["linear", "A_melhor", "B_compacta", "C_contraste"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax, m in zip(axes, nomes_m):
    curvas(ax, fin[f"{m}_seed0"], f"{m} ({fin[f'{m}_seed0']['n_params']:,} parâm.)".replace(",", "."))
    ax.set_xlabel("Época")
axes[0].set_ylabel("RMSE (pontos)"); axes[0].legend()
lo = min(min(h["train_rmse"] for h in fin[f"{m}_seed0"]["history"]) for m in nomes_m)
axes[0].set_ylim(lo - 1, lo + 6)
fig.suptitle("Curvas no treino completo (semente 0)", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); plots.save(fig, "final_curvas")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
t23 = agg[(agg["split"] == "test23") & (agg["modelo"] != "media_treino")].set_index("modelo").reindex(nomes_m)
for ax, (col, rot) in zip(axes, [("n_params", "Parâmetros treináveis"), ("treino_s", "Tempo de treino (s, média de 3 sementes)")]):
    for i, (m, r) in enumerate(t23.iterrows()):
        ax.errorbar(r[col], r["rmse"], yerr=r["rmse_dp"], fmt="o", color=SERIES[i], markersize=7, label=m, capsize=0)
    ax.set_xscale("log"); ax.set_xlabel(rot); ax.minorticks_off()
    ticks = [3e4, 1e5, 3e5, 1e6] if col == "n_params" else [10, 30, 100, 300, 1000]
    ax.set_xticks(ticks, [f"{t/1000:.0f} mil" if col == "n_params" and t < 1e6 else ("1 mi" if col == "n_params" else f"{t:.0f}") for t in ticks])
axes[0].set_ylabel("RMSE teste-2023 (pontos)"); axes[0].legend()
fig.suptitle("Custo × efetividade no teste 2023", x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout(); plots.save(fig, "final_custo_efetividade")
"""),
    md("## 5. Análise de erros (melhor modelo na validação, teste-2023)"),
    code("""
melhor = agg[(agg["split"] == "val") & agg["modelo"].isin(["A_melhor", "B_compacta", "C_contraste"])].sort_values("rmse")["modelo"].iloc[0]
print("modelo analisado:", melhor)
meta = load_meta("test23")
pred = pd.read_parquet(C.OUTPUTS / "predicoes_test23.parquet")
meta["pred"] = pred[melhor].values
meta["pred_linear"] = pred["linear"].values
meta["erro"] = meta["pred"] - meta[C.TARGET]
print(f"RMSE {np.sqrt((meta['erro']**2).mean()):.2f} | MAE {meta['erro'].abs().mean():.2f} | viés {meta['erro'].mean():.2f}")
print(f"faixa das predições: {meta['pred'].min():.0f}–{meta['pred'].max():.0f} (real: {meta[C.TARGET].min():.0f}–{meta[C.TARGET].max():.0f})")
print(f"desvio das predições {meta['pred'].std():.1f} vs desvio real {meta[C.TARGET].std():.1f}")
resumo_erro = {"modelo": melhor, "rmse": float(np.sqrt((meta['erro']**2).mean())), "mae": float(meta['erro'].abs().mean()),
               "vies": float(meta['erro'].mean()), "std_pred": float(meta['pred'].std()), "std_real": float(meta[C.TARGET].std()),
               "pred_min": float(meta['pred'].min()), "pred_max": float(meta['pred'].max()),
               "frac_erro_abs_lt_50": float((meta['erro'].abs() < 50).mean()), "frac_erro_abs_gt_100": float((meta['erro'].abs() > 100).mean())}
_ = (C.TAB_DIR / "analise_erros.json").write_text(json.dumps(resumo_erro, indent=1))
"""),
    code("""
fig, ax = plt.subplots(figsize=(6.4, 5.4))
hb = ax.hexbin(meta[C.TARGET], meta["pred"], gridsize=70, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", plots.BLUE_RAMP),
               mincnt=1, bins="log", linewidths=0)
lims = [250, 860]; ax.plot(lims, lims, color=INK_2, linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_xlabel("Nota média real"); ax.set_ylabel("Nota média prevista")
fig.colorbar(hb, ax=ax, label="participantes (escala log)")
ax.set_title("Predito × real (2023): o modelo comprime as notas para o centro"); plots.save(fig, "erro_hexbin")
"""),
    code("""
meta["decil_real"] = pd.qcut(meta[C.TARGET], 10, labels=[f"D{i}" for i in range(1, 11)])
dec = meta.groupby("decil_real", observed=True).agg(nota_real=(C.TARGET, "mean"), prevista=("pred", "mean"),
        vies=("erro", "mean"), mae=("erro", lambda e: e.abs().mean()))
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(dec))
ax.plot(x, dec["nota_real"], marker="o", color=SERIES[0], label="real (média do decil)")
ax.plot(x, dec["prevista"], marker="o", color=SERIES[1], label="prevista (média)")
ax.set_xticks(x, dec.index); ax.set_xlabel("Decil da nota real"); ax.set_ylabel("Nota média (pontos)"); ax.legend()
ax.set_title("Regressão à média: superestima quem vai mal e subestima quem vai bem"); plots.save(fig, "erro_decis")
dec.to_csv(C.TAB_DIR / "erro_decis.csv"); dec.round(1)
"""),
    code("""
def por_grupo(col, mapa=None):
    g = meta.groupby(col, observed=True).agg(n=("erro", "size"), real=(C.TARGET, "mean"), vies=("erro", "mean"),
                                             rmse=("erro", lambda e: np.sqrt((e**2).mean())))
    if mapa: g.index = [mapa.get(i, i) for i in g.index]
    return g
grupos = {
  "Tipo de escola": por_grupo("TP_ESCOLA", {1: "não respondeu", 2: "pública", 3: "privada"}),
  "Sexo (não usado no modelo)": por_grupo("TP_SEXO", {"F": "feminino", "M": "masculino"}),
  "Cor/raça (não usada no modelo)": por_grupo("TP_COR_RACA", {0: "não declarado", 1: "branca", 2: "preta", 3: "parda", 4: "amarela", 5: "indígena", 6: "sem informação"}),
  "Treineiro": por_grupo("IN_TREINEIRO", {0: "não", 1: "sim"}),
  "Redação": meta.assign(r0=np.where(meta["NU_NOTA_REDACAO"] == 0, "nota 0", "nota > 0")).pipe(lambda m: m.groupby("r0").agg(
        n=("erro", "size"), real=(C.TARGET, "mean"), vies=("erro", "mean"), rmse=("erro", lambda e: np.sqrt((e**2).mean())))),
}
gtab = pd.concat(grupos, names=["grupo", "categoria"])
gtab.to_csv(C.TAB_DIR / "erro_por_grupo.csv"); gtab.round(1)
"""),
    code("""
uf = por_grupo("SG_UF_PROVA").sort_values("vies"); uf.to_csv(C.TAB_DIR / "erro_uf.csv")
fig, ax = plt.subplots(figsize=(11, 3.4))
cores = [SERIES[0] if v >= 0 else SERIES[1] for v in uf["vies"]]
ax.bar(uf.index, uf["vies"], width=0.6, color=cores)
ax.axhline(0, color=MUTED, linewidth=1); ax.set_ylabel("Viés médio (previsto − real)")
ax.set_title("Viés por UF em 2023 (azul = superestima, laranja = subestima)"); plots.save(fig, "erro_uf")
"""),
    code("""
renda = por_grupo("Q006"); renda.to_csv(C.TAB_DIR / "erro_renda.csv")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(range(len(renda)), renda["rmse"], marker="o", color=SERIES[0], label="RMSE")
ax.plot(range(len(renda)), renda["vies"], marker="o", color=SERIES[1], label="viés")
ax.axhline(0, color=MUTED, linewidth=1)
ax.set_xticks(range(len(renda)), renda.index); ax.set_xlabel("Faixa de renda (Q006)"); ax.set_ylabel("Pontos"); ax.legend()
ax.set_title("Erro por faixa de renda (2023)"); plots.save(fig, "erro_renda")
"""),
]

if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    nb = nbf.v4.new_notebook(cells=EDA)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_DIR / "01_analise_exploratoria.ipynb")
    nb2 = nbf.v4.new_notebook(cells=RES)
    nb2.metadata["kernelspec"] = nb.metadata["kernelspec"]
    nbf.write(nb2, NB_DIR / "02_experimentos_e_resultados.ipynb")
    print("ok")
