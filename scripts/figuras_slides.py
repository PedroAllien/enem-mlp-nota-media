"""Figuras com fontes grandes para os slides da apresentação (outputs/figures/slides/)."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enem import config as C, plots
from enem.data import load_year
from enem.plots import INK_2, MUTED, SERIES
from enem.runs import load_results

plots.setup()
plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.labelsize": 20, "xtick.labelsize": 18,
                     "ytick.labelsize": 18, "legend.fontsize": 18, "lines.linewidth": 3, "savefig.dpi": 200})
# paleta dos slides: petróleo/ocre sobre creme (validada para daltonismo)
SERIES = ["#0A9396", "#C27C0E"]
CREME = "#F7F3EA"
plt.rcParams.update({"axes.prop_cycle": plt.cycler(color=SERIES), "figure.facecolor": CREME,
                     "axes.facecolor": CREME, "savefig.facecolor": CREME})
OUT = C.FIG_DIR / "slides"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


# 1) nota média por faixa de renda (2022)
d = load_year(2022)
letras = list("ABCDEFGHIJKLMNOPQ")
g = d.groupby("Q006")[C.TARGET].mean().reindex(letras)
fig, ax = plt.subplots(figsize=(8, 6.2))
ax.plot(range(17), g.values, color=SERIES[0], marker="o", markersize=9, markeredgecolor=CREME, markeredgewidth=2)
ax.axhline(d[C.TARGET].mean(), color=MUTED, linewidth=1.5)
ax.text(16.3, d[C.TARGET].mean() - 4, "média geral", color=INK_2, fontsize=16, ha="right", va="top")
ax.text(0, g.iloc[0] - 9, f"{g.iloc[0]:.0f}", ha="center", va="top", fontsize=20, color=INK_2)
ax.text(16, g.iloc[-1] + 9, f"{g.iloc[-1]:.0f}", ha="center", va="bottom", fontsize=20, color=INK_2)
ax.set_xticks([0, 4, 8, 12, 16], ["sem renda", "≈2 SM", "≈5 SM", "≈9 SM", ">20 SM"])
ax.set_ylabel("Nota média (pontos)"); ax.set_ylim(450, 680)
ax.set_xlabel("Renda familiar (Q006)")
save(fig, "renda")

# 2) curvas de capacidade (500 mil): 1x32 x 6x512
cap = {r["name"]: r for r in load_results("capacidade")}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, (k, rot) in zip(axes, [("n500k_d1_w32", "1 camada × 32 neurônios"), ("n500k_d6_w512", "6 camadas × 512 neurônios")]):
    h = pd.DataFrame(cap[k]["history"])
    ax.plot(h["epoch"], h["train_rmse"], color=SERIES[0], label="treino")
    ax.plot(h["epoch"], h["val_rmse"], color=SERIES[1], label="validação")
    ax.set_title(rot, fontsize=20); ax.set_xlabel("Época")
    ax.text(h["epoch"].iloc[-1], h["val_rmse"].iloc[-1] + 1.2, f"{h['val_rmse'].iloc[-1]:.0f}", ha="right", va="bottom", fontsize=20, color=INK_2)
    ax.text(h["epoch"].iloc[-1], h["train_rmse"].iloc[-1] - 1.2, f"{h['train_rmse'].iloc[-1]:.0f}", ha="right", va="top", fontsize=20, color=INK_2)
axes[0].set_ylabel("RMSE (pontos)"); axes[1].legend(loc="lower left"); axes[0].set_ylim(30, 90)
fig.tight_layout()
save(fig, "capacidade")

# 3) regularização: RMSE do modelo devolvido, duas escalas
reg = {r["name"]: r for r in load_results("regularizacao")}
var = [("sem_regularizacao", "sem regularização"), ("batchnorm", "BatchNorm"), ("dropout_0.5", "dropout 0,5"),
       ("l2_1e-3", "L2 (λ = 10⁻³)"), ("early_stopping", "early stopping")]
fig, ax = plt.subplots(figsize=(8, 6.2))
y = np.arange(len(var))
for i, (esc, rot) in enumerate([("n20k", "20 mil exemplos"), ("n500k", "500 mil exemplos")]):
    vals = [reg[f"{esc}_{v}"]["val"]["rmse"] for v, _ in var]
    ax.scatter(vals, y + (i - 0.5) * 0.3, s=180, color=SERIES[i], edgecolor=CREME, linewidth=2, zorder=3, label=rot)
ax.set_yticks(y, [r for _, r in var]); ax.invert_yaxis()
ax.set_xlabel("RMSE de validação do modelo final (pontos)"); ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
ax.grid(axis="y", visible=False)
save(fig, "regularizacao")

# 4) importância dos hiperparâmetros (Optuna, fANOVA) — recalcula do estudo
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name="enem_mlp", storage=f"sqlite:///{C.OUTPUTS / 'optuna.db'}")
imp = pd.Series(optuna.importance.get_param_importances(
    study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=C.SEED))).sort_values()
nomes = {"l2": "penalidade L2", "dropout": "dropout", "emb_municipio": "embedding município", "normalizacao": "normalização",
         "lr": "taxa de aprendizado", "ativacao": "ativação", "batch_size": "tamanho do lote", "largura": "largura",
         "n_camadas": "nº de camadas", "formato": "formato"}
fig, ax = plt.subplots(figsize=(8, 6.2))
cores = [SERIES[1] if k in ("largura", "n_camadas", "formato") else SERIES[0] for k in imp.index]
ax.barh([nomes[k] for k in imp.index], imp.values, height=0.6, color=cores)
for i, v in enumerate(imp.values):
    ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=18, color=INK_2)
ax.set_xlabel("Importância (fANOVA)"); ax.grid(axis="y", visible=False); ax.set_xlim(0, max(imp.values) * 1.2)
save(fig, "importancia")

# 5) regressão à média por decil (teste 2023)
dec = pd.read_csv(C.TAB_DIR / "erro_decis.csv")
fig, ax = plt.subplots(figsize=(8, 6.2))
x = np.arange(len(dec))
ax.plot(x, dec["nota_real"], marker="o", markersize=9, color=SERIES[0], label="nota real")
ax.plot(x, dec["prevista"], marker="o", markersize=9, color=SERIES[1], label="nota prevista")
ax.set_xticks(x, [f"{i}" for i in range(1, 11)]); ax.set_xlabel("Decil da nota real (1 = piores, 10 = melhores)")
ax.set_ylabel("Nota média (pontos)"); ax.legend(loc="upper left")
ax.annotate(f"+{dec['vies'].iloc[0]:.0f}", (0, (dec["nota_real"].iloc[0] + dec["prevista"].iloc[0]) / 2), xytext=(14, 0),
            textcoords="offset points", fontsize=20, color=INK_2, va="center")
ax.annotate(f"{dec['vies'].iloc[-1]:.0f}", (9, (dec["nota_real"].iloc[-1] + dec["prevista"].iloc[-1]) / 2), xytext=(-14, 0),
            textcoords="offset points", fontsize=20, color=INK_2, va="center", ha="right")
save(fig, "decis")
print(json.dumps({"renda_A": round(g.iloc[0]), "renda_Q": round(g.iloc[-1])}))

# 6) curvas das três arquiteturas finais (semente 0) — gap final é a média das 3 sementes
fin = {r["name"]: r for r in load_results("finais")}
modelos = [("A_melhor", "A · melhor (2×658)"), ("B_compacta", "B · compacta (1×173)"), ("C_contraste", "C · profunda (380→47)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, (m, rot) in zip(axes, modelos):
    h = pd.DataFrame(fin[f"{m}_seed0"]["history"])
    gap = np.mean([fin[f"{m}_seed{s}"]["history"][-1]["val_rmse"] - fin[f"{m}_seed{s}"]["history"][-1]["train_rmse"] for s in range(3)])
    ax.plot(h["epoch"], h["train_rmse"], color=SERIES[0], label="treino")
    ax.plot(h["epoch"], h["val_rmse"], color=SERIES[1], label="validação")
    ax.set_title(rot, fontsize=22); ax.set_xlabel("Época")
    ax.text(0.97, 0.93, f"gap final {gap:.1f}".replace(".", ","), transform=ax.transAxes, ha="right", va="top", fontsize=20, color=INK_2)
axes[0].set_ylabel("RMSE (pontos)"); axes[0].set_ylim(66.5, 70.5); axes[0].legend(loc="lower left")
fig.tight_layout()
save(fig, "finais")
