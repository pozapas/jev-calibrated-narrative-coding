"""F1 -- the method framework, as an editable draw.io diagram.

    python src/s19_framework_drawio.py --export

Visual language. This follows the architecture-diagram idiom now standard in machine-learning
papers rather than the boxes-of-bullets idiom common in transportation ones: flat pastel
blocks with no strokes, very little text, and the objects that flow between stages drawn as
token strips instead of described in prose. Two earlier drafts of this figure were built the
other way and both had the same defect -- a reader had to read eleven paragraphs to learn
what a diagram should show at a glance.

Three things the idiom buys that are not decoration:

  the token strips carry real values. The probability strip is shaded by actual returned
  probabilities, so the two-decimal grid and its floor are visible rather than asserted, and
  the review strip fills exactly the measured share of flagged records a human must open.

  the frozen and fitted marks carry the paper's thesis. The model is never trained; the only
  fitted object in the whole pipeline is the one-dimensional recalibration map, and putting a
  snowflake on the first and a flame on the second says that in two glyphs.

  the legend does the work a caption otherwise does, so the blocks can stay nearly wordless.

Every number, every shade and every fill fraction is bound from analysis.json,
cost_model.json and gold_analysis.json at build time, as numbers.tex binds the prose.

Output: outputs/figures/F1_framework.drawio  (+ .pdf / .png via --export)
"""
from __future__ import annotations
import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "s09_figures"))
import style as S

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
OUT = ROOT / "paper1" / "outputs" / "figures"
DRAWIO_EXE = Path(r"C:/Program Files/draw.io/draw.io.exe")

PAGE_W, PAGE_H = 1460, 600
FONT = "Helvetica"

# Flat fills, no strokes, in the paper's own hues. The idiom reads as a soft block of colour,
# so the tint is heavier than it would be for a bordered box.
FILL = {"grey": "#E7EAEC", "blue": "#D6E6F4", "deep": "#BBD6EE", "orange": "#FBE7C8",
        "ink": "#DFE3E6", "red": "#F8DCCE", "cream": "#FAF3E2"}


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def shade(p: float) -> str:
    """A probability as a fill: pale at the grid floor, saturated at the top of the range."""
    lo, hi = np.array([214, 230, 244]), np.array([0, 90, 158])
    c = lo + (hi - lo) * float(np.clip(p, 0, 1)) ** 0.62
    return "#{:02X}{:02X}{:02X}".format(*(int(round(v)) for v in c))


class Diagram:
    def __init__(self) -> None:
        self.cells: list[str] = []

    def raw(self, cid, style, value, x, y, w, h) -> None:
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="{esc(value)}" '
            f'vertex="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f'        </mxCell>')

    def block(self, cid, x, y, w, h, fill, text, size=25, sub=(), subsize=13) -> None:
        """A flat pastel block. No stroke: the colour is the boundary.

        `sub` is up to two short lines. They sit against the block's lower edge rather than
        being measured from its centre, so a one-line and a two-line title both clear them.
        """
        if isinstance(sub, str):
            sub = (sub,) if sub else ()
        self.raw(f"{cid}_b", f"rounded=1;arcSize=16;whiteSpace=wrap;html=1;fillColor={fill};"
                             f"strokeColor=none;shadow=0;", "", x, y, w, h)
        self.raw(f"{cid}_t",
                 f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;"
                 f"fontSize={size};fontFamily={FONT};fontColor={S.INK};fontStyle=1;",
                 text, x, y, w, h - 20 * len(sub) - 10)
        for i, line in enumerate(sub):
            self.raw(f"{cid}_s{i}",
                     f"text;html=1;align=center;verticalAlign=top;whiteSpace=wrap;"
                     f"fontSize={subsize};fontFamily={FONT};fontColor=#5C6B78;",
                     line, x + 4, y + h - 20 * (len(sub) - i) - 8, w - 8, 20)

    def text(self, cid, t, x, y, w, size=15, colour="#5C6B78", bold=False, align="center",
             h=None) -> None:
        self.raw(cid, f"text;html=1;align={align};verticalAlign=top;whiteSpace=wrap;"
                      f"fontSize={size};fontFamily={FONT};fontColor={colour};"
                      f"fontStyle={1 if bold else 0};", t, x, y, w, h or size + 24)

    def cell(self, cid, x, y, s, fill, stroke="none") -> None:
        self.raw(cid, f"rounded=1;arcSize=28;whiteSpace=wrap;html=1;fillColor={fill};"
                      f"strokeColor={stroke};shadow=0;", "", x, y, s, s)

    def strip(self, cid, x, y, vals, cols, size=20, gap=4, fill=None) -> tuple:
        """A row-major grid of token squares. `fill` overrides the per-value shading."""
        for i, v in enumerate(vals):
            r, c = divmod(i, cols)
            self.cell(f"{cid}{i}", x + c * (size + gap), y + r * (size + gap), size,
                      fill(v) if fill else shade(v))
        rows = -(-len(vals) // cols)
        return cols * (size + gap) - gap, rows * (size + gap) - gap

    def dashed(self, cid, x, y, w, h) -> None:
        self.raw(cid, "rounded=1;arcSize=10;fillColor=none;strokeColor=#9AA7B4;"
                      "strokeWidth=1.2;dashed=1;dashPattern=5 4;", "", x, y, w, h)

    def arrow(self, cid, x1, y1, x2, y2, colour="#3D4A55", width=2.2, dashed=False) -> None:
        style = (f"endArrow=blockThin;endFill=1;html=1;rounded=0;strokeWidth={width};"
                 f"strokeColor={colour};" + ("dashed=1;dashPattern=5 4;" if dashed else ""))
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="" edge="1">\n'
            '          <mxGeometry relative="1" as="geometry">\n'
            f'            <mxPoint x="{x1}" y="{y1}" as="sourcePoint" />\n'
            f'            <mxPoint x="{x2}" y="{y2}" as="targetPoint" />\n'
            '          </mxGeometry>\n        </mxCell>')

    def elbow(self, cid, pts, colour="#3D4A55", width=2.2, dashed=False) -> None:
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=blockThin;"
                 f"endFill=1;strokeWidth={width};strokeColor={colour};"
                 + ("dashed=1;dashPattern=5 4;" if dashed else ""))
        mid = "\n".join(f'              <mxPoint x="{a}" y="{b}" />' for a, b in pts[1:-1])
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="" edge="1">\n'
            '          <mxGeometry relative="1" as="geometry">\n'
            f'            <mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint" />\n'
            f'            <mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint" />\n'
            + (f'            <Array as="points">\n{mid}\n            </Array>\n' if mid else "")
            + '          </mxGeometry>\n        </mxCell>')

    def xml(self) -> str:
        return ('<mxfile host="app.diagrams.net">\n  <diagram id="jev-f1" name="Framework">\n'
                f'    <mxGraphModel dx="1400" dy="800" grid="0" gridSize="10" guides="1" '
                f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{PAGE_W}" pageHeight="{PAGE_H}" math="0" shadow="0">\n'
                '      <root>\n        <mxCell id="0" />\n'
                '        <mxCell id="1" parent="0" />\n' + "\n".join(self.cells)
                + '\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n')


def build() -> Diagram:
    a = json.loads((DATA / "analysis.json").read_text())
    cm = json.loads((DATA / "cost_model.json").read_text())
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    c, run, P = a["corpus"], a["run"], G["pooled"]
    q27 = cm["per_question_set"]["27"]
    b90 = P["review_budget_90"]

    # Real returned probabilities, so the strip shows the model's actual output grid rather
    # than an invented gradient. Quantiles rather than a random draw so it is reproducible.
    gp = pd.read_csv(DATA / "gold" / "gold_labels.csv")
    gp = gp[gp.p.notna()].p.to_numpy(float)
    raw_p = list(np.quantile(gp, np.linspace(0.06, 0.97, 5)))

    schema = json.loads((ROOT / "schemas" / "crash_factors_v1_1.json").read_text())
    n_gated = len(schema.get("gates", {}))

    # The calibrated strip shows the map the paper actually recommends deploying, fitted
    # here on the same gold labels s17 uses. The previous draft scaled the raw values by a
    # constant, which looked plausible and was invented; a figure that claims its strips
    # carry real values has to mean it.
    from sklearn.linear_model import LogisticRegression
    import s07_metrics as M
    gy = pd.read_csv(DATA / "gold" / "gold_labels.csv")
    gy = gy[gy.y.notna() & gy.p.notna()]
    platt = LogisticRegression().fit(M._logit(gy.p.to_numpy(float)).reshape(-1, 1),
                                     gy.y.to_numpy(float),
                                     sample_weight=gy.ht_weight.to_numpy(float))

    d = Diagram()
    d.text("title", "Calibrated Narrative Coding", 44, 22, 700, size=30,
           colour=S.INK, bold=True, align="left")

    BY, BH = 92, 156
    d.block("n1", 44, BY, 186, BH, FILL["grey"], "Narrative",
            sub=(f"{c['n_kept_gt_40'] / 1e6:.2f} M analysed",
                 "over 40 chars, PII screened"))
    d.block("n2", 280, BY, 196, BH, FILL["blue"], "Typed<br>Schema",
            sub=("16 presence, 8 choice, 3 score",
                 f"{n_gated} detail questions gated"))
    d.block("n3", 526, BY, 186, BH, FILL["deep"], "Jev 1.13",
            sub=("one parallel pass, no text out",
                 "p on a two-decimal grid"))

    # the model's output: one token per variable, shaded by its probability
    sw, sh = d.strip("tp", 746, BY + 12, raw_p, cols=1, size=24, gap=5)
    d.dashed("tp_g", 740, BY + 6, sw + 12, sh + 12)
    d.text("tp_l", "p", 740, BY + BH + 20, sw + 12, size=17, colour="#2B6CA3", bold=True)

    d.block("n4", 836, BY, 206, BH, FILL["orange"], "Recalibration",
            size=22, sub=("Platt, fitted out of fold",
                          "isotonic is the achievable floor"))

    cal_p = list(platt.predict_proba(
        M._logit(np.array(raw_p)).reshape(-1, 1))[:, 1])
    sw2, _ = d.strip("cp", 1090, BY + 12, cal_p, cols=1, size=24, gap=5)
    d.dashed("cp_g", 1084, BY + 6, sw2 + 12, sh + 12)
    d.text("cp_l", "calibrated p", 1056, BY + BH + 20, 120, size=17, colour="#C07C1E",
           bold=True)

    # the deliverable: a queue in which only the measured share is opened by a person
    NQ, COLS = 48, 12
    n_open = int(round(b90 * NQ))
    qw, qh = d.strip("q", 1202, BY + 30, [1] * NQ, cols=COLS, size=17, gap=4,
                     fill=lambda _i: FILL["ink"])
    for i in range(n_open):                       # overdraw the reviewed ones
        r, col = divmod(i, COLS)
        d.cell(f"qo{i}", 1202 + col * 21, BY + 30 + r * 21, 17, S.INK)
    d.text("q_l", "Review queue", 1186, BY + BH + 20, qw + 30, size=17, colour=S.INK,
           bold=True)
    d.text("q_s", f"H(0.90) = {100 * b90:.1f}% of flags", 1176, BY + BH + 42, qw + 50,
           size=14)

    for i, (x1, x2) in enumerate([(230, 276), (476, 522), (712, 736),
                                  (806, 832), (1042, 1080), (1160, 1198)]):
        d.arrow(f"a{i}", x1, BY + BH // 2, x2, BY + BH // 2)

    # the 27 questions, as tokens, with the seven gated ones left hollow
    toks = [1] * (27 - n_gated) + [0] * n_gated
    d.strip("sq", 286, BY + BH + 62, toks, cols=9, size=15, gap=4,
            fill=lambda v: FILL["deep"] if v else "#EFF4F9")
    d.text("sq_l", "presence gates detail; gated answers stay uninterpreted",
           246, BY + BH + 114, 320, size=13)
    d.dashed("stage_g", 36, BY - 26, 684, BH + 22)
    d.text("stage_l",
           f"Stage 1 screens 10 presence questions over the random sample and sizes the Stage 2 frame: "
           f"{run['n_stage2_random']:,} random plus rare-class enrichment, every record carrying its inclusion probability",
           38, BY + BH + 14, 688, size=13, align="left")

    # ------------------------------------------------------------ the two references
    RY = 322
    d.text("ref_h", "THE AUDIT DECIDES WHAT p MEANS", 596, RY - 32, 420, size=16,
           colour=S.INK, bold=True, align="left")
    d.block("r1", 596, RY, 196, 54, FILL["grey"], "Coded CRIS fields", size=16)
    d.strip("r1t", 812, RY + 17, [1] * 6, cols=6, size=18, gap=4,
            fill=lambda _i: "#9AA7B4")
    d.text("r1x", f"agreement, {run['n_stage2_random']:,} crashes", 940, RY + 14, 250,
           size=14, align="left")

    d.block("r2", 596, RY + 74, 196, 54, FILL["orange"], "Human gold set", size=16)
    d.strip("r2t", 812, RY + 91, [1] * 6, cols=6, size=18, gap=4,
            fill=lambda _i: S.ORANGE)
    d.text("r2x", f"accuracy, {G['n_usable']:,} labels, 3 blinded coders", 940, RY + 88,
           280, size=14, align="left")

    d.elbow("e_r1", [(792, RY + 27), (960, RY + 27), (960, BY + BH)], "#9AA7B4", 2.0,
            dashed=True)
    d.elbow("e_r2", [(792, RY + 101), (912, RY + 101), (912, BY + BH)], S.ORANGE, 2.2)
    d.text("e_note",
           "compared against both; fitted on the human labels alone, because the coded "
           "fields flatter the model",
           596, RY + 140, 560, size=14, align="left")

    # ------------------------------------------------------------ fixed vs fitted
    d.text("f1", "&#10052;", 662, BY + 2, 40, size=26, colour="#5FA8D3")
    d.text("f2", "&#128293;", 998, BY + 2, 40, size=26, colour="#E06C2A")

    # ------------------------------------------------------------ legend
    LY = 502
    items = [(FILL["deep"], "typed question"), ("#2B6CA3", "model probability"),
             ("#9AA7B4", "coded CRIS label"), (S.ORANGE, "human label"),
             (S.INK, "record a person opens")]
    x = 44
    for i, (col, lab) in enumerate(items):
        d.cell(f"lg{i}", x, LY + 3, 16, col)
        d.text(f"lgt{i}", lab, x + 24, LY, 190, size=14, align="left")
        x += 26 + 9 * len(lab) + 22
    d.text("lg_f", "&#10052; fixed, never trained&#160;&#160;&#160;"
                   "&#128293; the only fitted object in the pipeline",
           44, LY + 30, 640, size=14, align="left")
    full = c["n_kept_gt_40"] * q27["cost_per_1k_narratives_usd"] / 1000
    d.text("lg_c",
           f"${q27['cost_per_1k_narratives_usd']:.3f} per 1,000&#160;&#160;&#183;&#160;&#160;{run['p50_latency_s']:.2f} s median&#160;&#160;&#183;&#160;&#160;"
           f"${run['total_cost_usd']:.0f} to code the {run['n_stage2']:,} evaluated&#160;&#160;&#183;&#160;&#160;"
           f"${full:,.0f} to code all {c['n_kept_gt_40'] / 1e6:.2f} M",
           780, LY + 30, 640, size=14, align="right")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
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
