"""F1 -- Framework overview (§3).

Four lanes: NARRATIVE -> TYPED DECISIONS (presence Nouls gating Choices/Scores) ->
PROBABILISTIC TABLE -> AUDIT LOOP (reliability, risk-coverage, review queue).
Arrows are annotated with the objects they carry: p_iv, pi_iv, c_iv, tau, H(r).

The probability readout uses real values from the v1.1 schema-validation run so the figure
shows the shape of what the model actually returns. The narrative glyph is synthetic text
written for this figure -- no real CRIS narrative is reproduced.

The main axes spans the whole figure (add_axes([0,0,1,1])) so that axes coordinates and
figure coordinates coincide and the inset mini-plots land where the lane layout puts them.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

sys.path.insert(0, str(Path(__file__).parent))
import style as S

GLYPH = ["DRIVER 1 STATED SHE FELT",
         "DIZZY AND BLACKED OUT",
         "BRIEFLY BEFORE THE VEHICLE",
         "LEFT THE ROADWAY. PASSENGER",
         "WAS [NAME], 7 MONTHS",
         "PREGNANT, AND WAS",
         "TRANSPORTED AS A PRECAUTION."]

NOULS = [("medical_episode", 0.94), ("preg_mentioned", 0.98),
         ("transported", 0.96), ("alcohol_involved", 0.02)]
CHOICE_ID = "medical_type"
CHOICE = [("loss_of_consc.", 0.71), ("other_medical", 0.18),
          ("seizure", 0.07), ("cardiac", 0.04)]
SCORE_ID = "injury_narr"
SCORE = [0.05, 0.22, 0.68, 0.05]

# lane x, width, title
LANES = [(0.004, 0.205, "1. NARRATIVE"),
         (0.219, 0.335, "2. TYPED DECISIONS"),
         (0.564, 0.185, "3. PROBABILISTIC TABLE"),
         (0.759, 0.237, "4. AUDIT LOOP")]


def box(ax, x, y, w, h, fc, ec, lw=0.7, r=0.008):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def bar(ax, x, y, wfull, h, frac, color, label, fs=4.8):
    ax.add_patch(Rectangle((x, y - h / 2), wfull * frac, h, fc=color, ec="none", zorder=3))
    ax.add_patch(Rectangle((x, y - h / 2), wfull, h, fc="none", ec="#C4C4CC", lw=0.35, zorder=4))
    ax.text(x + wfull + 0.004, y, label, fontsize=fs, va="center", ha="left",
            family="monospace", color="#444444", zorder=5)


def arrow(ax, p0, p1, label=None, color="#444444", rad=0.0, lw=1.0, fs=6.0, dy=0.016):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7.5, lw=lw,
                                 color=color, zorder=6,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + dy, label, ha="center",
                va="bottom", fontsize=fs, color=color, zorder=7)


def mini_reliability(fig):
    a = fig.add_axes([0.806, 0.615, 0.125, 0.205])
    xs = np.linspace(0.03, 0.97, 9)
    a.plot([0, 1], [0, 1], ls=(0, (2, 2)), lw=0.6, color="#AAAAAA")
    a.plot(xs, np.clip(xs + 0.055 * np.sin(3.1 * xs) - 0.012, 0, 1), "-o",
           color=S.BLUE, ms=1.7, lw=0.8, label="human")
    a.plot(xs, np.clip(xs * 0.80 + 0.035, 0, 1), "-o", color=S.GREY, ms=1.5, lw=0.7,
           alpha=0.85, label="coded")
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    a.set_title("calibration", fontsize=5.8, pad=2)
    a.set_xlabel("predicted $p$", fontsize=5.2, labelpad=1.5)
    a.set_ylabel("observed", fontsize=5.2, labelpad=1.5)
    a.legend(fontsize=4.4, loc="upper left", handlelength=1.0, borderpad=0.15,
             labelspacing=0.15, borderaxespad=0.15)


def mini_riskcov(fig):
    a = fig.add_axes([0.806, 0.285, 0.125, 0.205])
    c = np.linspace(0.04, 1, 80)
    a.plot(c, 0.004 + 0.10 * c ** 3.4, color=S.BLUE, lw=0.9)
    a.axhline(0.05, color=S.ORANGE, lw=0.7, ls=(0, (2, 1.6)))
    a.axvline(0.80, color=S.ORANGE, lw=0.5, ls=(0, (1, 1.6)))
    a.set_xlim(0, 1); a.set_ylim(0, 0.13); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    a.set_title("risk-coverage", fontsize=5.8, pad=2)
    a.set_xlabel("coverage $C$", fontsize=5.2, labelpad=1.5)
    a.set_ylabel("risk $R$", fontsize=5.2, labelpad=1.5)
    a.text(0.04, 0.058, "target $r$", fontsize=4.6, color=S.ORANGE, va="bottom")
    a.annotate("", xy=(0.80, 0.012), xytext=(1.0, 0.012),
               arrowprops=dict(arrowstyle="<->", lw=0.5, color=S.ORANGE))
    a.text(0.90, 0.020, "$H(r)$", fontsize=4.8, color=S.ORANGE, ha="center")


def main() -> None:
    S.setup()
    fig = plt.figure(figsize=(S.W2, 3.55))
    ax = fig.add_axes([0, 0, 1, 1])          # axes coords == figure coords
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    for x, w, t in LANES:
        ax.add_patch(Rectangle((x, 0.035), w, 0.875, fc="#FAFAFC", ec="#E3E3E9",
                               lw=0.6, zorder=0))
        ax.text(x + w / 2, 0.925, t, ha="center", va="bottom", fontsize=7.0,
                fontweight="bold", color="#2F2F2F")

    # ---------------------------------------------------------------- lane 1
    lx, lw_ = LANES[0][0], LANES[0][1]
    box(ax, lx + 0.010, 0.520, lw_ - 0.020, 0.330, "#FFFFFF", "#BFBFC8")
    ax.text(lx + lw_ / 2, 0.828, "police crash narrative", ha="center", va="top",
            fontsize=6.0, style="italic", color="#6A6A6A")
    for i, line in enumerate(GLYPH):
        ax.text(lx + 0.019, 0.782 - i * 0.031, line, ha="left", va="center",
                fontsize=4.5, family="monospace", color="#2A2A2A")
    ax.text(lx + lw_ / 2, 0.470, "5.02 M narratives\nTexas CRIS 2017-2025",
            ha="center", va="top", fontsize=6.2, color="#555555", linespacing=1.45)
    ax.text(lx + lw_ / 2, 0.365, 'state = {"narrative": ...}', ha="center", va="center",
            fontsize=5.2, family="monospace", color=S.BLUE)
    ax.text(lx + lw_ / 2, 0.300, "names and phones already\nreplaced by placeholders;\n"
                                 "second-pass redaction applied",
            ha="center", va="top", fontsize=5.2, color="#777777", linespacing=1.45)

    # ---------------------------------------------------------------- lane 2
    nx, nw = 0.229, 0.150
    ax.text(nx, 0.880, "presence (Noul)", fontsize=6.2, color="#3A3A3A", va="top")
    for i, (name, p) in enumerate(NOULS):
        y = 0.822 - i * 0.052
        box(ax, nx, y - 0.021, nw, 0.042, "#FFFFFF", "#CDCDD5", lw=0.55, r=0.006)
        ax.text(nx + 0.006, y, name, fontsize=4.9, va="center", family="monospace",
                color="#333333", zorder=5)
        bar(ax, nx + 0.098, y, 0.030, 0.018, p, S.BLUE if p > 0.5 else S.SKY, f"{p:.2f}", fs=4.4)

    gx, gy = 0.406, 0.758
    ax.add_patch(Polygon([[gx, gy + 0.036], [gx + 0.027, gy], [gx, gy - 0.036],
                          [gx - 0.027, gy]], fc="#FFF5E2", ec=S.ORANGE, lw=0.8, zorder=3))
    ax.text(gx, gy + 0.004, r"$p_{g}>\tau_g$", ha="center", va="center", fontsize=5.0, zorder=4)
    arrow(ax, (nx + nw + 0.002, 0.812), (gx - 0.026, gy + 0.012), rad=-0.2, lw=0.8)
    ax.text(gx, gy - 0.052, "gate", ha="center", fontsize=5.6, color=S.ORANGE, va="top")

    cx, cw_ = 0.385, 0.155
    box(ax, cx, 0.470, cw_, 0.178, "#FFFFFF", "#CDCDD5", lw=0.55, r=0.006)
    ax.text(cx + 0.006, 0.632, CHOICE_ID + "   (Choice)", fontsize=4.9, va="center",
            family="monospace", color="#333333")
    for j, (opt, q) in enumerate(CHOICE):
        yy = 0.598 - j * 0.030
        ax.text(cx + 0.008, yy, opt, fontsize=4.4, va="center", color="#555555")
        bar(ax, cx + 0.095, yy, 0.026, 0.014, q, S.BLUE if j == 0 else S.SKY, f"{q:.2f}", fs=4.1)

    box(ax, cx, 0.250, cw_, 0.178, "#FFFFFF", "#CDCDD5", lw=0.55, r=0.006)
    ax.text(cx + 0.006, 0.412, SCORE_ID + "     (Score)", fontsize=4.9, va="center",
            family="monospace", color="#333333")
    for j, q in enumerate(SCORE):
        yy = 0.378 - j * 0.030
        ax.text(cx + 0.008, yy, f"level {j}", fontsize=4.4, va="center", color="#555555")
        bar(ax, cx + 0.095, yy, 0.026, 0.014, q, S.BLUE if q == max(SCORE) else S.SKY,
            f"{q:.2f}", fs=4.1)
    arrow(ax, (gx, gy - 0.072), (cx + cw_ / 2, 0.654), rad=0.0, lw=0.8)

    ax.text(nx + 0.010, 0.560, "no text generation:\nthe model selects,\nnever writes",
            ha="left", va="top", fontsize=5.6, color=S.BLUE, linespacing=1.5)
    ax.text(nx + 0.010, 0.440, "27 questions\n3,683 input tokens\n$0.155 / 1,000\nnarratives",
            ha="left", va="top", fontsize=5.6, color="#555555", linespacing=1.5)
    ax.text(nx + 0.010, 0.290, "presence first,\ndetail second", ha="left", va="top",
            fontsize=5.6, color=S.ORANGE, linespacing=1.5)

    # ---------------------------------------------------------------- lane 3
    tx = 0.573
    cols, cw, rh = ["med", "preg", "alc", "inj"], 0.036, 0.048
    vals = [[0.94, 0.98, 0.02, 2], [0.03, 0.01, 0.88, 1],
            [0.11, 0.00, 0.06, 0], [0.61, 0.02, 0.45, 3]]
    ax.text(tx, 0.880, "one row per narrative", fontsize=6.2, color="#3A3A3A", va="top")
    for j, c in enumerate(cols):
        ax.text(tx + 0.024 + j * cw + cw / 2, 0.818, c, ha="center", fontsize=5.0,
                color="#555555", family="monospace")
    for i, row in enumerate(vals):
        y = 0.800 - i * rh
        ax.text(tx + 0.002, y - rh / 2, f"n{i+1}", fontsize=4.8, va="center",
                family="monospace", color="#888888")
        for j, v in enumerate(row):
            xx = tx + 0.024 + j * cw
            shade = v if j < 3 else v / 3
            ax.add_patch(Rectangle((xx, y - rh), cw, rh, fc=S.BLUE,
                                   alpha=0.07 + 0.75 * shade, ec="#FFFFFF", lw=0.7, zorder=2))
            ax.text(xx + cw / 2, y - rh / 2, f"{v:.2f}" if j < 3 else f"{v:.0f}",
                    ha="center", va="center", fontsize=4.7, zorder=3, family="monospace",
                    color="white" if shade > 0.55 else "#333333")
    ax.text(tx, 0.560, "probabilities,\nnot hard labels", ha="left", va="top",
            fontsize=6.0, color=S.BLUE, linespacing=1.45)
    ax.text(tx, 0.470, "analysts threshold,\nweight, or route\nto review", ha="left",
            va="top", fontsize=5.6, color="#555555", linespacing=1.5)
    ax.text(tx, 0.340, "every cell carries\nits own uncertainty", ha="left", va="top",
            fontsize=5.6, color="#777777", linespacing=1.5)

    # ---------------------------------------------------------------- lane 4
    mini_reliability(fig)
    mini_riskcov(fig)
    box(ax, 0.768, 0.075, 0.220, 0.130, "#FFF5E2", S.ORANGE, lw=0.8)
    ax.text(0.878, 0.172, "human-review budget", ha="center", va="center", fontsize=6.2,
            color="#8A5A00", fontweight="bold")
    ax.text(0.878, 0.132, r"$H(r) \;=\; 1 - C^{*}(r)$", ha="center", va="center",
            fontsize=6.4, color="#8A5A00")
    ax.text(0.878, 0.098, "narratives per year to check", ha="center", va="center",
            fontsize=5.4, color="#8A5A00")

    # ---------------------------------------------------------------- inter-lane arrows
    arrow(ax, (0.190, 0.660), (0.226, 0.660), r"$n_i$", lw=1.1, fs=6.4)
    arrow(ax, (0.545, 0.660), (0.570, 0.660), lw=1.1)
    ax.text(0.5575, 0.690, r"$p_{iv},\ \pi_{iv},\ c_{iv}$", ha="center", va="bottom",
            fontsize=6.0, color="#444444")
    arrow(ax, (0.744, 0.862), (0.800, 0.862), "gold set", lw=1.1, fs=5.8, dy=0.010)
    ax.add_patch(FancyArrowPatch((0.800, 0.140), (0.700, 0.300), arrowstyle="-|>",
                                 mutation_scale=7.5, lw=1.0, color=S.ORANGE, zorder=6,
                                 connectionstyle="arc3,rad=0.30"))
    ax.text(0.735, 0.185, r"$\tau$", fontsize=7.5, color=S.ORANGE, ha="center", va="center")

    S.save(fig, "F1_framework")


if __name__ == "__main__":
    main()
