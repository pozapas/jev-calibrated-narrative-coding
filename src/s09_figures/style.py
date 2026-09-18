"""Shared publication figure system: white, aligned, colourblind-safe, 300 dpi PDF+PNG.

Palette convention from §3: blue = data/model, orange = human/decision. Greys carry
reference/coded-field series so the "reference-noise gap" in F4 reads without colour.
seaborn is not installed and is not used.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures"

# Okabe-Ito, colourblind-safe
BLUE = "#0072B2"    # data / model (Jev)
ORANGE = "#E69F00"  # human / decision
GREEN = "#009E73"   # keyword baseline
RED = "#D55E00"     # LLM baseline / errors
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#7F7F7F"    # coded-field reference
LIGHT = "#D9E1E5"
INK = "#24323A"
MUTED = "#66757F"
WHITE = "#FFFFFF"

# Use these values in every panel.  Keeping them here prevents small visual drift
# as figures are revised one by one.
LABEL_PAD = 6.0
TITLE_PAD = 7.0

CYCLE = [BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY, YELLOW]
METHOD = {"jev": BLUE, "human": ORANGE, "keyword": GREEN, "llm": RED, "coded": GREY}


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.labelpad": LABEL_PAD,
        "axes.titlepad": TITLE_PAD,
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "grid.linewidth": 0.4,
        "grid.color": LIGHT,
        "grid.alpha": 0.72,
        "grid.linestyle": ":",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType so the PDF is editable/vector
        "ps.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def _sentence_case(label: str) -> str:
    """Capitalise the first alphabetic character without breaking TeX labels."""
    for i, char in enumerate(label):
        if char.isalpha():
            return label[:i] + char.upper() + label[i + 1:]
    return label


def finish(fig, *, grid: str = "y") -> None:
    """Apply the shared final-pass rules before the figure layout is calculated.

    This keeps x/y label padding, title spacing, white panels, tick spacing, and
    restrained reference grids identical across the paper.
    """
    fig.patch.set_facecolor(WHITE)
    for ax in fig.get_axes():
        if not ax.get_visible():
            continue
        ax.set_facecolor(WHITE)
        ax.set_xlabel(_sentence_case(ax.get_xlabel()), labelpad=LABEL_PAD)
        ax.set_ylabel(_sentence_case(ax.get_ylabel()), labelpad=LABEL_PAD)
        ax.tick_params(pad=3.0)
        ax.set_axisbelow(True)
        for side in ("left", "bottom"):
            spine = ax.spines.get(side)
            if spine is not None:
                spine.set_color("#65737E")
        if grid:
            ax.grid(axis=grid, zorder=0)
    fig.align_labels()


# Elsevier single/double column widths in inches
W1, W15, W2 = 3.54, 5.51, 7.48


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def panel_label(ax, s: str, dx: float = -0.12, dy: float = 1.06) -> None:
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


def thousands(ax, axis: str = "y") -> None:
    f = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)
