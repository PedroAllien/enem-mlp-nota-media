"""Estilo único para todas as figuras do relatório (matplotlib, fundo claro)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config as C

# Paleta categórica validada para daltonismo (usar sempre nesta ordem).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "sans-serif", "font.size": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "patch.linewidth": 0,
    })


def save(fig, name: str):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.png")
    return fig
