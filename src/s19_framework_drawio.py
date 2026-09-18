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

# Token colours are separated by HUE, not by lightness of one hue. The first draft used
# pale blue for a typed question, dark blue for a probability and grey for a coded
# label: the first two then read as one thing in two amounts, and the first and third
# were hard to tell apart at token size. Each category now has its own hue, and the one
# continuous encoding (probability) is the only thing that varies in lightness.
TOK = {"ask": "#9B8AC4",        # a question asked of every narrative
       "gated": "#E4DEF2",      # a detail question its gate has not opened
       "coded": "#6E7B87",      # a coded CRIS label
       "human": "#E69F00",      # a human label
       "open": "#1C2B36",       # a record a person opens
       "queue": "#DCE1E5"}      # a record nobody opens


def esc(s: str) -> str:
    return html.escape(s, quote=True)


P_FLOOR = 1e-3


def shade(p: float) -> str:
    """A probability as a fill, on a LOG scale from 0.001 to 1.

    Linear lightness cannot show this pipeline's own subject matter. Most of the mass
    sits below p = 0.05, and recalibration moves values within that region -- 0.24 to
    0.056, 0.01 to under 0.001 -- so on a linear ramp the raw and calibrated strips came
    out looking identical and the figure failed at the one comparison it exists to make.
    A log ramp spends its contrast where the data and the correction actually are.
    """
    lo, hi = np.array([233, 240, 246]), np.array([11, 79, 130])
    f = np.log10(max(float(p), P_FLOOR) / P_FLOOR) / np.log10(1 / P_FLOOR)
    c = lo + (hi - lo) * float(np.clip(f, 0, 1))
    return "#{:02X}{:02X}{:02X}".format(*(int(round(v)) for v in c))


def pfmt(p: float) -> str:
    """Print the value beside its cell: colour alone cannot be read to two decimals."""
    return "&lt;0.01" if p < 0.005 else f"{p:.2f}"


class Diagram:
    def __init__(self) -> None:
        self.cells: list[str] = []

    def raw(self, cid, style, value, x, y, w, h) -> None:
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="{esc(value)}" '
            f'vertex="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f'        </mxCell>')

    def block(self, cid, x, y, w, h, fill, text, size=25, sub=(), subsize=13,
              title_h=None) -> None:
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
                 text, x, y + (4 if title_h else 0), w,
                 title_h or (h - 20 * len(sub) - 10))
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

    def strip(self, cid, x, y, vals, cols, size=20, gap=4, fill=None,
              stroke_pale=False) -> tuple:
        """A row-major grid of token squares. `fill` overrides the per-value shading.

        `stroke_pale` outlines the near-white tokens, which otherwise read as empty space
        rather than as a token that exists but is switched off.
        """
        for i, v in enumerate(vals):
            r, c = divmod(i, cols)
            col = fill(v) if fill else shade(v)
            edge = "#B9C3CD" if (stroke_pale and not v) else "none"
            self.cell(f"{cid}{i}", x + c * (size + gap), y + r * (size + gap), size,
                      col, stroke=edge)
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
    d.block("n1", 44, BY, 186, BH, FILL["grey"], "Crash Narratives",
            sub=(f"{c['n_kept_gt_40'] / 1e6:.2f} M analysed",
                 "over 40 chars, PII screened"))
    d.block("n2", 280, BY, 196, BH, FILL["blue"], "Typed Schema", title_h=42,
            sub=("16 presence, 8 choice, 3 score",
                 f"{n_gated} gated, left uninterpreted"))
    # The paper's claim is about System One models as an interface; Jev is the instance it
    # happens to audit, and Section 6 says the audit transfers where the model does not.
    d.block("n3", 526, BY, 186, BH, FILL["deep"], "System One<br>Model",
            sub=("typed answers, no text out",
                 "Jev 1.13"))

    # the model's output: one token per variable, shaded by its probability
    sw, sh = d.strip("tp", 746, BY + 12, raw_p, cols=1, size=24, gap=5)
    for i, v in enumerate(raw_p):
        d.text(f"tpv{i}", pfmt(v), 772, BY + 14 + i * 29, 46, size=11,
               colour="#5C6B78", align="left", h=20)
    d.dashed("tp_g", 740, BY + 6, sw + 12, sh + 12)
    d.text("tp_l", "p", 740, BY + BH + 20, sw + 12, size=17, colour="#2B6CA3", bold=True)

    d.block("n4", 836, BY, 206, BH, FILL["orange"], "Recalibration",
            size=22, sub=("Platt, fitted out of fold",
                          "isotonic is the achievable floor"))

    cal_p = list(platt.predict_proba(
        M._logit(np.array(raw_p)).reshape(-1, 1))[:, 1])
    sw2, _ = d.strip("cp", 1090, BY + 12, cal_p, cols=1, size=24, gap=5)
    for i, v in enumerate(cal_p):
        d.text(f"cpv{i}", pfmt(v), 1116, BY + 14 + i * 29, 46, size=11,
               colour="#5C6B78", align="left", h=20)
    d.dashed("cp_g", 1084, BY + 6, sw2 + 12, sh + 12)
    d.text("cp_l", "calibrated p", 1056, BY + BH + 20, 120, size=17, colour="#C07C1E",
           bold=True)

    # the deliverable: a queue in which only the measured share is opened by a person
    NQ, COLS = 48, 12
    n_open = int(round(b90 * NQ))
    qw, qh = d.strip("q", 1202, BY + 30, [1] * NQ, cols=COLS, size=17, gap=4,
                     fill=lambda _i: TOK["queue"])
    for i in range(n_open):                       # overdraw the reviewed ones
        r, col = divmod(i, COLS)
        d.cell(f"qo{i}", 1202 + col * 21, BY + 30 + r * 21, 17, TOK["open"])
    d.text("q_l", "Review budget", 1186, BY + BH + 20, qw + 30, size=17, colour=S.INK,
           bold=True)
    d.text("q_s", f"H(0.90) = {100 * b90:.1f}% of flags", 1176, BY + BH + 42, qw + 50,
           size=14)

    for i, (x1, x2) in enumerate([(230, 276), (476, 522), (712, 736),
                                  (806, 832), (1042, 1080), (1160, 1198)]):
        d.arrow(f"a{i}", x1, BY + BH // 2, x2, BY + BH // 2)

    # the 27 questions, as tokens, with the seven gated ones left hollow
    toks = [1] * (27 - n_gated) + [0] * n_gated
    d.strip("sq", 294, BY + 52, toks, cols=9, size=15, gap=4,
            fill=lambda v: TOK["ask"] if v else TOK["gated"], stroke_pale=True)
    d.dashed("stage_g", 36, BY - 26, 684, BH + 22)
    d.text("stage_l",
           f"Stage 1 screens 10 questions and sizes the Stage 2 frame: "
           f"{run['n_stage2_random']:,} random plus rare-class enrichment, "
           f"inverse-probability weighted.",
           39, BY + BH + 28, 498, size=13, align="left")

    # ------------------------------------------------------------ the two references
    RY = 322
    d.text("ref_h", "THE AUDIT DECIDES WHAT p MEANS", 596, RY - 32, 420, size=16,
           colour=S.INK, bold=True, align="left")
    d.block("r1", 596, RY, 196, 54, FILL["grey"], "Coded CRIS fields", size=16)
    d.strip("r1t", 812, RY + 17, [1] * 6, cols=6, size=18, gap=4,
            fill=lambda _i: TOK["coded"])
    d.text("r1x", f"agreement, {run['n_stage2_random']:,} crashes", 940, RY + 14, 250,
           size=14, align="left")

    d.block("r2", 596, RY + 74, 196, 54, FILL["orange"], "Human gold set", size=16)
    d.strip("r2t", 812, RY + 91, [1] * 6, cols=6, size=18, gap=4,
            fill=lambda _i: TOK["human"])
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
    items = [(TOK["ask"], "question asked"), (TOK["gated"], "gate not opened"),
             (TOK["coded"], "coded CRIS label"), (TOK["human"], "human label"),
             (TOK["open"], "flagged record, checked"),
             (TOK["queue"], "flagged record, accepted")]
    x = 44
    for i, (col, lab) in enumerate(items):
        d.cell(f"lg{i}", x, LY + 3, 16, col,
               stroke="#B9C3CD" if col in (TOK["gated"], TOK["queue"]) else "none")
        d.text(f"lgt{i}", lab, x + 23, LY, 200, size=14, align="left")
        x += 25 + 8 * len(lab) + 20
    # Probability is the one continuous encoding, so its key is a ramp with its end
    # points named, not a single swatch a reader would take for another category.
    for i, v in enumerate([0.001, 0.01, 0.05, 0.2, 0.6, 1.0]):
        d.cell(f"lgr{i}", x + i * 18, LY + 3, 16, shade(v))
    d.text("lgr_a", "p", x - 16, LY, 16, size=13, align="right")
    d.text("lgr_b", "log scale, 0.001 to 1", x + 116, LY, 220,
           size=14, align="left")
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
    ap.add_argument("--force", action="store_true",
                    help="overwrite a diagram that was edited by hand in draw.io")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "F1_framework.drawio"

    # draw.io stamps host="Electron" when the desktop app saves. This script writes
    # host="app.diagrams.net", so that stamp means a person has since moved things by hand,
    # and regenerating would silently throw their layout away. Re-export from the edited file
    # instead, or pass --force if the hand edits really are meant to go.
    if f.exists() and "Electron" in f.read_text(encoding="utf-8")[:200] and not args.force:
        print(f"{f.name} was edited in draw.io after it was generated; refusing to overwrite.")
        print("  re-export instead:  draw.io --export --format pdf --crop "
              "--output F1_framework.pdf F1_framework.drawio")
        print("  or discard those edits:  rerun with --force")
        return

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
