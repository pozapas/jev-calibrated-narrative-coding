"""F1 -- the method framework, emitted as an editable draw.io diagram.

    python src/s19_framework_drawio.py --export

Design. The first draft of this figure was a grid of boxes each holding five bullets, which
is the common idiom and the wrong one: it asks the reader to read a paragraph in eleven
places and encodes nothing visually. This version carries the same content in three parts:

  a SPINE of six nodes, one line each, that is the pipeline and nothing else;
  an AUDIT LOOP hanging beneath it, because the audit modifies the probability rather than
    extending the flow, and the shape should say so;
  three EVIDENCE PANELS -- real plots from the real analysis, one per headline finding.

The panels are why the figure can be short. "Cost is governed by schema size" costs a bullet
and convinces nobody; the same claim as seven measured points on an axis takes less room and
is harder to argue with.

Every number and every panel is generated from analysis.json, cost_model.json,
gold_analysis.json and frontier_analysis.json at build time, exactly as numbers.tex binds the
prose, so the figure cannot fall out of step with the paper.

Colour follows style.py, which already fixes blue = the typed model, orange = human
judgement, grey = coded fields, red = the baselines held apart. A reader arriving from a
later figure reads the same hues as the same things.

Output: outputs/figures/F1_framework.drawio  (+ .pdf / .png via --export)
"""
from __future__ import annotations
import argparse
import base64
import html
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "s09_figures"))
import style as S

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
OUT = ROOT / "paper1" / "outputs" / "figures"
DRAWIO_EXE = Path(r"C:/Program Files/draw.io/draw.io.exe")

PAGE_W, PAGE_H = 1380, 560
FONT = "Helvetica"
PANEL_DPI = 320


def tint(hexcolor: str, keep: float) -> str:
    """Blend a palette colour toward white, so a fill can never drift from its stroke."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda v: round(v * keep + 255 * (1 - keep))
    return f"#{f(r):02X}{f(g):02X}{f(b):02X}"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


# ------------------------------------------------------------------ evidence panels
def _png64(fig) -> str:
    """Render a matplotlib figure to the base64 PNG form draw.io embeds inline."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=PANEL_DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bare(ax, keep=("bottom",)) -> None:
    for name, sp in ax.spines.items():
        if name in keep:
            sp.set_color("#8894A0")
            sp.set_linewidth(0.7)
        else:
            sp.set_visible(False)
    ax.tick_params(labelsize=6.0, length=2, pad=1.5)


def panel_cost(cm: dict) -> str:
    """Input tokens against schema size: seven measured points rather than an assertion."""
    ks = sorted(cm["per_question_set"], key=int)
    x = np.array([int(k) for k in ks])
    y = np.array([cm["per_question_set"][k]["mean_tokens"] for k in ks])
    fig, ax = plt.subplots(figsize=(2.7, 1.12))
    ax.plot(x, y, "-", color=S.BLUE, lw=1.6, zorder=2)
    ax.scatter(x, y, s=20, color=S.BLUE, zorder=3, lw=0)
    b = cm["intercept_model"]["beta_tokens_per_question"]
    ax.annotate(f"+{b:.0f} tokens per question", xy=(1.0, max(y) * 1.06),
                ha="left", va="top", fontsize=6.4, color=S.INK)
    ax.set_xlabel("questions in the schema", fontsize=6.4, labelpad=1.5)
    ax.set_ylabel("input tokens", fontsize=6.4, labelpad=1.5)
    ax.set_xlim(0, 29)
    ax.set_ylim(0, max(y) * 1.18)
    _bare(ax, keep=("bottom", "left"))
    return _png64(fig)


def panel_reference(a: dict, G: dict) -> str:
    """Per-variable ECE against each reference, paired.

    A dumbbell is the right form because the claim is about the gap and its direction: every
    variable's orange dot sits right of its grey one. Two separate bar charts would leave the
    reader to establish that by eye.
    """
    rows = []
    for v, r in G["per_variable"].items():
        cod = a["vs_coded_fields"].get(v, {}).get("ece_vs_coded")
        if cod:
            rows.append((cod, r["ece"]))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(2.7, 1.12))
    for i, (cod, gold) in enumerate(rows):
        ax.plot([cod, gold], [i, i], "-", color="#B9C3CD", lw=1.1, zorder=1)
    ax.scatter([r[0] for r in rows], range(len(rows)), s=15, color=S.GREY, zorder=3, lw=0,
               label="vs. coded fields")
    ax.scatter([r[1] for r in rows], range(len(rows)), s=15, color=S.ORANGE, zorder=3, lw=0,
               label="vs. human labels")
    ax.set_yticks([])
    ax.set_xlabel("expected calibration error", fontsize=6.4, labelpad=1.5)
    ax.set_ylim(-1.4, len(rows) - 0.2)
    ax.legend(fontsize=5.9, frameon=False, loc="lower right", handletextpad=0.2,
              borderpad=0.05, labelspacing=0.2, handlelength=0.8)
    _bare(ax)
    return _png64(fig)


def panel_budget(G: dict) -> str:
    """Per-variable review budget: the paper's question, answered variable by variable."""
    vals = np.array(sorted(r["review_budget_90"]
                           for r in G["per_variable"].values())) * 100
    fig, ax = plt.subplots(figsize=(2.7, 1.12))
    free = vals <= 1e-9
    ax.barh(range(len(vals)), np.maximum(vals, 0.8), height=0.74, edgecolor="none",
            color=[S.LIGHT if f else S.BLUE for f in free])
    pooled = 100 * G["pooled"]["review_budget_90"]
    ax.axvline(pooled, color=S.ORANGE, lw=1.2, ls=(0, (3, 2)))
    ax.annotate(f"pooled {pooled:.1f}%", xy=(pooled, len(vals) * 0.52), xytext=(4, 0),
                textcoords="offset points", fontsize=6.2, color=S.ORANGE, va="center")
    ax.annotate(f"{int(free.sum())} of {len(vals)} need\nno review at all", xy=(0, 0.0),
                xytext=(3, -1), textcoords="offset points", fontsize=6.2, color=S.INK,
                va="bottom", linespacing=1.2)
    ax.set_yticks([])
    ax.set_xlabel("% of flagged records a human opens", fontsize=6.4, labelpad=1.5)
    ax.set_ylim(-1.5, len(vals) - 0.2)
    _bare(ax)
    return _png64(fig)


# ------------------------------------------------------------------ diagram emitters
class Diagram:
    def __init__(self) -> None:
        self.cells: list[str] = []

    def raw(self, cid: str, style: str, value: str, x, y, w, h) -> None:
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="{esc(value)}" '
            f'vertex="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f'        </mxCell>')

    def node(self, cid: str, x, y, w, h, colour: str, title: str, sub: str,
             dashed: bool = False, strong: bool = False) -> None:
        d = "dashed=1;dashPattern=8 6;" if dashed else ""
        self.raw(f"{cid}_box",
                 f"rounded=1;whiteSpace=wrap;html=1;fillColor={tint(colour, 0.12)};"
                 f"strokeColor={colour};strokeWidth={2.6 if strong else 1.8};arcSize=10;"
                 f"shadow=1;{d}", "", x, y, w, h)
        self.raw(f"{cid}_t",
                 f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=21;"
                 f"fontFamily={FONT};fontColor={S.INK};fontStyle=1;",
                 title, x + 5, y + 8, w - 10, 26)
        self.raw(f"{cid}_s",
                 f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=16;"
                 f"fontFamily={FONT};fontColor={colour};",
                 sub, x + 5, y + 35, w - 10, 22)

    def label(self, cid: str, text: str, x, y, w, size=16, colour=None, bold=False,
              align="left") -> None:
        self.raw(cid, f"text;html=1;align={align};verticalAlign=top;whiteSpace=wrap;"
                      f"fontSize={size};fontFamily={FONT};"
                      f"fontColor={colour or S.INK};fontStyle={1 if bold else 0};",
                 text, x, y, w, size + 26)

    def image(self, cid: str, b64: str, x, y, w, h) -> None:
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="shape=image;imageAspect=0;'
            f'html=1;image=data:image/png,{b64}" value="" vertex="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f'        </mxCell>')

    def arrow(self, cid: str, src: str, tgt: str, label: str = "", colour: str = S.BLUE,
              exit_xy=None, entry_xy=None, dashed=False, width=2.6, points=None) -> None:
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
                 f"html=1;strokeWidth={width};strokeColor={colour};endArrow=blockThin;"
                 f"endFill=1;fontSize=15;fontFamily={FONT};fontColor={S.INK};"
                 f"labelBackgroundColor=#FFFFFF;")
        if dashed:
            style += "dashed=1;dashPattern=6 4;"
        if exit_xy:
            style += f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
        if entry_xy:
            style += f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;"
        geo = '          <mxGeometry relative="1" as="geometry" />'
        if points:
            pts = "\n".join(f'              <mxPoint x="{a}" y="{b}" />' for a, b in points)
            geo = ('          <mxGeometry relative="1" as="geometry">\n'
                   f'            <Array as="points">\n{pts}\n            </Array>\n'
                   '          </mxGeometry>')
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="{esc(label)}" '
            f'edge="1" source="{src}" target="{tgt}">\n{geo}\n        </mxCell>')

    def xml(self) -> str:
        return ('<mxfile host="app.diagrams.net">\n'
                '  <diagram id="jev-framework" name="Framework">\n'
                f'    <mxGraphModel dx="1400" dy="800" grid="0" gridSize="10" guides="1" '
                f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{PAGE_W}" pageHeight="{PAGE_H}" math="0" shadow="0">\n'
                '      <root>\n        <mxCell id="0" />\n'
                '        <mxCell id="1" parent="0" />\n'
                + "\n".join(self.cells) +
                '\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')


# ------------------------------------------------------------------ the framework
def build() -> Diagram:
    a = json.loads((DATA / "analysis.json").read_text())
    cm = json.loads((DATA / "cost_model.json").read_text())
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    F = json.loads((DATA / "frontier" / "frontier_analysis.json").read_text())
    c, run, P = a["corpus"], a["run"], G["pooled"]
    q27 = cm["per_question_set"]["27"]
    ratio = F["cost"]["claude-sonnet-5_per_1k_usd"] / F["cost"]["jev_per_1k_usd"]
    fable = F["arms"].get("claude-fable-5-1", {}).get("pooled", {}).get("f1")
    b90 = 100 * P["review_budget_90"]

    d = Diagram()
    NW, NH, NY = 180, 74, 40
    # The wider gap before node 5 is where the audit loop returns into the spine.
    XS = [36, 252, 468, 684, 936, 1152]

    # ------------------------------------------------------------- the spine
    d.node("n1", XS[0], NY, NW, NH, S.GREY, "Narratives",
           f"{c['n_kept_gt_40'] / 1e6:.2f} M, 2017 to 2025")
    d.node("n2", XS[1], NY, NW, NH, S.BLUE, "Typed schema", "27 gated questions")
    d.node("n3", XS[2], NY, NW, NH, S.BLUE, "Jev 1.13", "no text generated")
    d.node("n4", XS[3], NY, NW, NH, S.BLUE, "Raw p", "grid 0.01 to 0.99")
    d.node("n5", XS[4], NY, NW, NH, S.ORANGE, "Calibrated p",
           f"ECE {G['recalibration']['none']['ece']:.3f} to "
           f"{G['recalibration']['isotonic']['ece']:.3f}")
    d.node("n6", XS[5], NY, NW, NH, S.INK, "Review queue",
           f"H(0.90) = {b90:.1f}% of flags", strong=True)
    for i in range(3):
        d.arrow(f"s{i}", f"n{i + 1}_box", f"n{i + 2}_box", "", S.BLUE, (1, 0.5), (0, 0.5))
    d.arrow("s5", "n5_box", "n6_box", "", S.ORANGE, (1, 0.5), (0, 0.5))
    d.label("spine_note",
            f"${q27['cost_per_1k_narratives_usd']:.3f} per 1,000 narratives &#160;&#183;&#160; "
            f"{run['p50_latency_s']:.2f} s median &#160;&#183;&#160; "
            f"${run['total_cost_usd']:.2f} for the whole run",
            XS[0], 118, 620, size=15, colour=S.MUTED)

    # ------------------------------------------------------------- the audit loop
    d.label("audit_hd", "THE AUDIT DECIDES WHAT p MEANS",
            XS[0], 190, 380, size=16, colour=S.INK, bold=True)
    d.node("r1", XS[0], 216, 236, 58, S.GREY, "Coded CRIS fields",
           f"agreement, {run['n_stage2_random']:,} crashes")
    d.node("r2", XS[0], 280, 236, 58, S.ORANGE, "Human gold set",
           f"accuracy, {G['n_usable']:,} labels")
    d.node("au", 324, 216, 250, 92, S.BLUE, "Calibration audit",
           f"z = {P['spiegelhalter_z']:.1f}: rejected")
    d.label("au_x", "The coded fields make the model look better calibrated than it is, "
            "for every variable. Recalibrating on the human labels is what makes p usable.",
            612, 228, 336, size=15, colour=S.MUTED)
    d.arrow("a1", "r1_box", "au_box", "", S.GREY, (1, 0.5), (0, 0.28), width=2.0)
    d.arrow("a2", "r2_box", "au_box", "", S.ORANGE, (1, 0.5), (0, 0.72), width=2.0)
    d.arrow("a3", "n4_box", "au_box", "raw p", S.BLUE, (0.5, 1), (0.42, 0),
            points=[(XS[3] + NW // 2, 150), (429, 150)], width=2.2)
    d.arrow("a4", "au_box", "n5_box", "recalibrated on the human labels", S.ORANGE,
            (0.78, 0), (0.5, 1), points=[(519, 172), (XS[4] + NW // 2, 172)],
            width=2.4)

    # ------------------------------------------------------------- baselines, held apart
    d.node("bl", 1000, 216, 344, 58, S.RED, "Baselines, held apart",
           "same narratives, same criteria", dashed=True)
    if fable:
        d.label("bl_x",
                f"A frontier model gains {fable - P['f1']:+.3f} F&#8321; at "
                f"{ratio:.0f}&#215; the price. A second gains nothing measurable. "
                f"Neither is calibrated.",
                1000, 280, 344, size=15, colour=S.MUTED)

    # ------------------------------------------------------------- evidence panels
    PW, PH, PY = 400, 152, 390
    PX = [36, 476, 916]
    d.label("ev_hd", "WHAT IT FOUND", PX[0], 352, 400, size=16, colour=S.INK, bold=True)
    for i, (cid, cap, b64) in enumerate([
            ("p1", "Cost is set by the schema, not the text", panel_cost(cm)),
            ("p2", "The reference decides the answer", panel_reference(a, G)),
            ("p3", "The budget, variable by variable", panel_budget(G))]):
        d.label(f"{cid}_c", cap, PX[i], 370, PW, size=15, colour=S.INK, bold=True)
        d.image(cid, b64, PX[i], PY, PW, PH)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="also render PDF and PNG")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "F1_framework.drawio"
    f.write_text(build().xml(), encoding="utf-8")
    print(f"wrote {f} ({f.stat().st_size / 1024:.0f} KB)")
    if args.export:
        if not DRAWIO_EXE.exists():
            print("draw.io not found; export by hand")
            return
        for ext in ("pdf", "png"):
            cmd = [str(DRAWIO_EXE), "--export", "--format", ext, "--crop",
                   "--output", str(OUT / f"F1_framework.{ext}"), str(f)]
            if ext == "png":
                cmd[2:2] = ["--scale", "2"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            print(f"  {ext}: {'ok' if (OUT / f'F1_framework.{ext}').exists() else 'FAILED'} "
                  f"{r.stderr.strip()[:110]}")


if __name__ == "__main__":
    main()
