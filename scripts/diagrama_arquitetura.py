"""Figura com o diagrama da arquitetura (para o relatório)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enem import plots
from enem.plots import INK, INK_2, MUTED, SERIES, BLUE_RAMP

plots.setup()
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")


def box(x, y, w, h, title, sub, face, edge=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge or face, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), title, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8, color=INK_2)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=0, shrinkB=0))


# entradas
box(0.1, 2.95, 2.6, 0.8, "29 categóricas → one-hot", "191 colunas (ausente = categoria)", BLUE_RAMP[0])
box(0.1, 1.75, 2.6, 0.8, "Q005 (moradores)", "z-score + indicador de ausente (2 col.)", BLUE_RAMP[0])
box(0.1, 0.55, 2.6, 0.8, "Município da prova", "índice → embedding (d = 0…32)", BLUE_RAMP[0])
box(3.3, 1.45, 1.2, 1.4, "concat", "193 + d", "#f0efec")
for y in (3.35, 2.15, 0.95):
    arrow(2.7, y, 3.3, 2.15)
# bloco oculto
ax.add_patch(FancyBboxPatch((5.0, 0.75), 3.6, 2.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor="none", edgecolor=SERIES[0], linewidth=1.4, linestyle="-"))
ax.text(6.8, 3.75, "camada oculta  × L   (L = 0…6)", ha="center", fontsize=9.5, color=INK, fontweight="bold")
steps = [("Linear (n neurônios)", BLUE_RAMP[1]), ("BatchNorm / LayerNorm / nenhuma", "#f0efec"),
         ("ReLU / GELU / SiLU", BLUE_RAMP[1]), ("Dropout (p)", "#f0efec")]
for i, (t, c) in enumerate(steps):
    box(5.25, 2.85 - i * 0.62, 3.1, 0.5, t, None, c)
arrow(4.5, 2.15, 5.0, 2.15)
box(9.1, 1.6, 1.25, 1.1, "Linear → 1", "saída", BLUE_RAMP[2])
arrow(8.6, 2.15, 9.1, 2.15)
box(10.75, 1.6, 1.15, 1.1, "nota média", "ŷ·σ + μ (pontos)", "#f0efec")
arrow(10.35, 2.15, 10.75, 2.15)
ax.text(6.8, 0.3, "perda: MSE no alvo padronizado  +  λ₁‖W‖₁ + λ₂‖W‖₂²  ·  otimizador Adam  ·  early stopping na validação",
        ha="center", fontsize=8.5, color=INK_2)
plots.save(fig, "arquitetura")
