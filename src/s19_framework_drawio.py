"""F1 -- the method framework, emitted as an editable draw.io diagram.

    python src/s19_framework_drawio.py

Why a generator rather than a hand-drawn file: every number in this figure is bound from
analysis.json, cost_model.json, gold_analysis.json and frontier_analysis.json at build time,
exactly as numbers.tex binds the prose. A framework figure is the one place a stale number is
least likely to be noticed and most damaging, because a reader takes it as the summary of the
whole study. Re-run this after any re-analysis and the figure follows.

The .drawio file is XML and remains fully editable in draw.io -- this script produces the
first draft of the layout, not a locked image.

Colour carries the paper's established semantics rather than decoration. style.py already
fixes blue = Jev, orange = human, green = keyword, red = frontier LLM, grey = coded fields,
and a reader arriving from Figure 4 or Figure 7 reads those hues as those things. Fills are
light tints derived from the same hex, so the mapping cannot drift.

Output: outputs/figures/F1_framework.drawio  (+ .pdf / .png via --export)
"""
from __future__ import annotations
import argparse
import re
import html
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "s09_figures"))
import style as S

ROOT = Path(r"D:/OneDrive - Texas State University/AIT/Papers/Jev")
DATA = ROOT / "paper1" / "data"
OUT = ROOT / "paper1" / "outputs" / "figures"
DRAWIO_EXE = Path(r"C:/Program Files/draw.io/draw.io.exe")

PAGE_W, PAGE_H = 1460, 880
FONT = "Helvetica"


# --------------------------------------------------------------------------- colour helpers
def tint(hexcolor: str, keep: float) -> str:
    """Blend a palette colour toward white. `keep` is the share of the original hue.

    Derived rather than hand-picked so that a change to style.py propagates here and the
    fill can never drift away from the stroke it is supposed to be a tint of.
    """
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda c: round(c * keep + 255 * (1 - keep))
    return f"#{f(r):02X}{f(g):02X}{f(b):02X}"


def esc(s: str) -> str:
    """draw.io holds XML-escaped HTML in the value attribute; html=1 renders it."""
    return html.escape(s, quote=True)


# --------------------------------------------------------------------------- cell emitters
class Diagram:
    def __init__(self) -> None:
        self.cells: list[str] = []

    def raw(self, cid: str, style: str, value: str, x, y, w, h, vertex=True) -> str:
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="{esc(value)}" '
            f'vertex="1">\n'
            f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />\n'
            f'        </mxCell>')
        return cid

    def band(self, cid: str, title: str, x, y, w, h) -> None:
        self.raw(cid, f"rounded=1;fillColor=none;strokeColor={S.MUTED};strokeWidth=1;"
                      f"dashed=1;dashPattern=8 4;arcSize=4;fontSize=21;fontFamily={FONT};"
                      f"fontColor={S.INK};fontStyle=1;verticalAlign=top;align=left;"
                      f"spacingLeft=12;spacingTop=5;",
                 title, x, y, w, h)

    def box(self, cid: str, x, y, w, h, colour: str, dashed: bool = False) -> None:
        d = "dashed=1;dashPattern=8 6;" if dashed else ""
        self.raw(f"{cid}_box",
                 f"rounded=1;whiteSpace=wrap;html=1;fillColor={tint(colour, 0.10)};"
                 f"strokeColor={colour};strokeWidth=1.8;arcSize=8;shadow=1;{d}",
                 "", x, y, w, h)

    def header(self, cid: str, text: str, x, y, w, colour: str) -> None:
        self.raw(f"{cid}_hdr",
                 f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=19;"
                 f"fontFamily={FONT};fontColor={S.INK};fontStyle=1;",
                 text, x + 8, y + 8, w - 16, 25)

    def sub(self, cid: str, text: str, x, y, w, colour: str) -> None:
        self.raw(f"{cid}_sub",
                 f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=15;"
                 f"fontFamily={FONT};fontColor={colour};fontStyle=2;",
                 text, x + 8, y + 33, w - 16, 20)

    def sep(self, cid: str, x, y, w, colour: str) -> None:
        self.raw(f"{cid}_sep", f"line;strokeWidth=1;strokeColor={colour};opacity=35;",
                 "", x + 12, y + 55, w - 24, 1)

    def bullets(self, cid: str, items: list[str], x, y, w) -> int:
        """Advance by the number of lines each bullet will actually occupy.

        A fixed row height silently overlaps the next bullet as soon as one wraps, which is
        invisible in the XML and obvious in the render. Wrapping is estimated from the box's
        inner width at 14px Helvetica; long bullets are also shortened at source, so this is
        a guard rather than the primary mechanism.
        """
        cpl = max(16, int((w - 24) / 8.1))
        yy = y + 60
        for i, t in enumerate(items):
            plain = re.sub(r"<[^>]+>|&#?\w+;", "x", t)
            lines = max(1, -(-len(plain) // cpl))
            h = 20 * lines + 2
            self.raw(f"{cid}_t{i}",
                     f"text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize=16;"
                     f"fontFamily={FONT};fontColor=#26343C;spacingLeft=8;",
                     t, x + 6, yy, w - 12, h)
            yy += h
        return yy

    def out(self, cid: str, text: str, x, y, w, colour: str, idx: int = 0) -> None:
        """The object a stage hands on. Bottom-anchored so these line up across the row."""
        self.raw(f"{cid}_out{idx}",
                 f"text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=16;"
                 f"fontFamily={FONT};fontColor={colour};fontStyle=1;",
                 text, x + 6, y, w - 12, 20)

    def arrow(self, cid: str, src: str, tgt: str, label: str = "", colour: str = S.MUTED,
              exit_xy: tuple | None = None, entry_xy: tuple | None = None,
              dashed: bool = False, points: list[tuple] | None = None) -> None:
        """`points` forces the route through explicit waypoints.

        Left to itself, draw.io's orthogonal router will happily draw an edge straight
        through an intervening box -- the reference-to-audit edge crossed the gold-set box,
        and the feedback edge crossed an entire band of text. Waypoints put those segments in
        the empty lanes the layout reserves for them.
        """
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
                 f"html=1;strokeWidth=2.2;strokeColor={colour};endArrow=blockThin;endFill=1;"
                 f"fontSize=14;fontFamily={FONT};fontColor={S.INK};"
                 f"labelBackgroundColor=#FFFFFF;")
        if dashed:
            style += "dashed=1;dashPattern=6 4;"
        if exit_xy:
            style += f"exitX={exit_xy[0]};exitY={exit_xy[1]};exitDx=0;exitDy=0;"
        if entry_xy:
            style += f"entryX={entry_xy[0]};entryY={entry_xy[1]};entryDx=0;entryDy=0;"
        geo = '          <mxGeometry relative="1" as="geometry" />'
        if points:
            pts = "\n".join(f'              <mxPoint x="{px}" y="{py}" />' for px, py in points)
            geo = ('          <mxGeometry relative="1" as="geometry">\n'
                   '            <Array as="points">\n'
                   f'{pts}\n'
                   '            </Array>\n'
                   '          </mxGeometry>')
        self.cells.append(
            f'        <mxCell id="{cid}" parent="1" style="{style}" value="{esc(label)}" '
            f'edge="1" source="{src}" target="{tgt}">\n'
            f'{geo}\n'
            f'        </mxCell>')

    def xml(self) -> str:
        body = "\n".join(self.cells)
        return (
            '<mxfile host="app.diagrams.net">\n'
            '  <diagram id="jev-framework" name="Framework">\n'
            f'    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
            f'pageWidth="{PAGE_W}" pageHeight="{PAGE_H}" math="0" shadow="0">\n'
            '      <root>\n'
            '        <mxCell id="0" />\n'
            '        <mxCell id="1" parent="0" />\n'
            f'{body}\n'
            '      </root>\n'
            '    </mxGraphModel>\n'
            '  </diagram>\n'
            '</mxfile>\n')


def panel(d: Diagram, cid: str, x, y, w, h, colour: str, header: str, sub: str,
          items: list[str], outs: list[str], dashed: bool = False) -> None:
    """One stage box: frame, title, rule, bullets, and the object it hands on."""
    d.box(cid, x, y, w, h, colour, dashed)
    d.header(cid, header, x, y, w, colour)
    d.sub(cid, sub, x, y, w, colour)
    d.sep(cid, x, y, w, colour)
    d.bullets(cid, items, x, y, w)
    for i, o in enumerate(reversed(outs)):
        d.out(cid, o, x, y + h - 26 - 21 * i, w, colour, idx=i)


# --------------------------------------------------------------------------- the framework
def build() -> Diagram:
    a = json.loads((DATA / "analysis.json").read_text())
    cm = json.loads((DATA / "cost_model.json").read_text())
    G = json.loads((DATA / "gold" / "gold_analysis.json").read_text())
    R = json.loads((DATA / "gold" / "gold_reliability.json").read_text())
    F = json.loads((DATA / "frontier" / "frontier_analysis.json").read_text())
    c, run = a["corpus"], a["run"]
    P, q27 = G["pooled"], cm["per_question_set"]["27"]

    ks = [v for v in R["pairwise_kappa_calibration"].values() if v is not None]
    ece_gap = sorted(G["per_variable"][v]["ece"] / a["vs_coded_fields"][v]["ece_vs_coded"]
                     for v in G["per_variable"]
                     if v in a["vs_coded_fields"] and a["vs_coded_fields"][v]["ece_vs_coded"])
    gap_med = ece_gap[len(ece_gap) // 2]
    b90 = 100 * P["review_budget_90"]
    n_free = sum(1 for r in G["per_variable"].values() if r["review_budget_90"] <= 1e-9)
    secs = sum(x["median_ms"] for x in R["per_rater_calibration"].values()) / \
        len(R["per_rater_calibration"]) / 1000.0
    hours = R["n_answers"] / R["n_raters"] * secs / 3600.0
    cost_ratio = F["cost"]["claude-sonnet-5_per_1k_usd"] / F["cost"]["jev_per_1k_usd"]
    arms = {k: v["pooled"]["f1"] for k, v in F["arms"].items()}
    fable = arms.get("claude-fable-5-1")
    gpt = arms.get("gpt-5.6-sol")

    d = Diagram()
    # Four columns on a 1440-unit canvas. The figure is placed as a rotated full-page float,
    # so its scale is set by the text height: a narrower canvas carrying larger type is what
    # puts the labels at a readable size in print. Each band keeps an empty lane at its foot
    # and the page keeps one at the right, so edges route through white space, never over a
    # box. Five columns at this canvas width left the labels at roughly 4pt on the page.
    XS = [36, 396, 756, 1116]
    BW, BH = 300, 226
    GAP12, GAP23 = 312, 602
    LANE2 = 522
    RIGHT = 1438

    # ---------------------------------------------------------------- stage 1
    d.band("band1", "1.  Typed decisions over the full corpus", 18, 22, 1424, 272)
    y1 = 58
    panel(d, "a1", XS[0], y1, BW, BH, S.GREY,
          "Corpus", "Texas CRIS, 2017 to 2025",
          [f"&#8226; {c['n_rows_total']:,} crash records",
           f"&#8226; {c['n_kept_gt_40']:,} over 40 characters",
           "&#8226; two-pass streaming, fixed seed",
           "&#8226; redaction, then a PII screen"],
          ["&#8594; no narrative leaves the pipeline"])
    panel(d, "a2", XS[1], y1, BW, BH, S.BLUE,
          "Gated typed schema", "27 questions per narrative",
          ["&#8226; 16 presence, 8 Choice, 3 Score",
           "&#8226; presence gates detail, &#964;<sub>g</sub> = 0.5",
           "&#8226; options selected, not generated",
           "&#8226; the criteria are the specification"],
          ["&#8594; a schema, not a prompt"])
    panel(d, "a3", XS[2], y1, BW, BH, S.BLUE,
          "Jev 1.13", "System One: typed, no text out",
          ["&#8226; one parallel pass per narrative",
           f"&#8226; {run['mean_input_tokens']:,.0f} tokens, "
           f"{run['p50_latency_s']:.2f} s median",
           "&#8226; probabilities on a 2-decimal grid",
           f"&#8226; cost set by schema size, R&#178; {q27['r2']:.2f}"],
          [f"&#8594; ${q27['cost_per_1k_narratives_usd']:.3f} per 1,000; "
           f"${run['total_cost_usd']:.2f} in total"])
    panel(d, "a4", XS[3], y1, BW, BH, S.BLUE,
          "Probabilistic table", "the deliverable, not a label",
          ["&#8226; a row per crash, a column per variable",
           "&#8226; probability kept, not thresholded",
           f"&#8226; {run['n_stage2']:,} coded, {run['n_stage2_random']:,} random",
           "&#8226; a resolution floor at p = 0.01"],
          ["&#8594; p<sub>iv</sub> per narrative and variable"])

    # ---------------------------------------------------------------- stage 2
    d.band("band2", "2.  Two references, one audit", 18, 312, 1424, 272)
    y2 = 348
    panel(d, "b1", XS[0], y2, BW, BH, S.GREY,
          "Reference A: coded fields", "large n, agreement only",
          [f"&#8226; {len(a['vs_coded_fields'])} variables with a usable field",
           f"&#8226; {run['n_stage2_random']:,} joined crashes",
           "&#8226; incomplete, in one direction",
           "&#8226; narrative-only is not an error"],
          ["&#8594; bounds what agreement can show"])
    panel(d, "b2", XS[1], y2, BW, BH, S.ORANGE,
          "Reference B: human gold set", "the accuracy reference",
          [f"&#8226; {G['n_pairs_labelled']:,} pairs, {R['n_raters']} blinded coders",
           f"&#8226; stratified on p, HT weighted",
           f"&#8226; &#954; {min(ks):.2f} to {max(ks):.2f}, {G['n_usable']:,} usable",
           "&#8226; disputes excluded, not judged"],
          ["&#8594; y<sub>iv</sub> from human judgement"])
    panel(d, "b3", XS[2], y2, BW, BH, S.BLUE,
          "Calibration audit", "per variable, then recalibrated",
          ["&#8226; ECE on the model's own grid",
           "&#8226; Cox slope, Spiegelhalter z",
           "&#8226; Platt, isotonic, split-half",
           f"&#8226; {hours:.1f} h per coder buys the fit"],
          [f"&#8594; ECE {P['ece']:.3f}, z = {P['spiegelhalter_z']:.1f}: rejected",
           f"&#8594; coded flatter {gap_med:.1f}&#215;; recal. to "
           f"{G['recalibration']['isotonic']['ece']:.3f}"])
    panel(d, "b4", XS[3], y2, BW, BH, S.RED,
          "Baselines, held apart", "same narratives and criteria",
          ["&#8226; keyword rules: off-sample, near free",
           f"&#8226; Fable 5.1 {fable:.3f}, GPT 5.6 Sol {gpt:.3f}"
           if fable and gpt else "&#8226; two frontier models",
           f"&#8226; typed model {P['f1']:.3f}, same labels",
           "&#8226; none of the three is calibrated"],
          [f"&#8594; frontier accuracy at {cost_ratio:.0f}&#215; the price"],
          dashed=True)

    # ---------------------------------------------------------------- stage 3
    d.band("band3", "3.  What a safety office actually receives", 18, 602, 1424, 246)
    y3 = 638
    CW = 440
    panel(d, "c1", 36, y3, CW, 196, S.INK,
          "Selective prediction: the review budget", "Section 5.5",
          ["&#8226; accuracy-based risk is degenerate when events are rare",
           "&#8226; coverage is defined over <i>flagged positives</i> instead",
           "&#8226; H(&#960;) is the share of flags a human opens"],
          [f"&#8594; H(0.90) = {b90:.1f}% of flags; {n_free} of "
           f"{P['n_variables']} variables need none"])
    panel(d, "c2", 510, y3, CW, 196, S.RED,
          "Where these must not be used", "stated limits",
          ["&#8226; E[p] is not a prevalence estimate below the grid floor",
           "&#8226; agreement with coded fields is not accuracy",
           "&#8226; three variables fall below F<sub>1</sub> 0.70 and are named"],
          ["&#8594; limits reported, not averaged away"])
    panel(d, "c3", 984, y3, CW, 196, S.ORANGE,
          "Delivered", "a decision rule, not a label",
          [f"&#8226; calibrated probabilities over "
           f"{c['n_kept_gt_40'] / 1e6:.2f} M narratives",
           "&#8226; a review queue priced in staff hours",
           "&#8226; the audit repeats whenever the model changes"],
          ["&#8594; how much checking this variable needs"])

    # ---------------------------------------------------------------- flow
    # Every arrow is labelled with the object it carries, so the figure doubles as a
    # data-flow specification rather than a picture of boxes.
    for i, (s_, t_) in enumerate([("a1", "a2"), ("a2", "a3"), ("a3", "a4")]):
        lbl = ["narratives", "27 questions", "p<sub>iv</sub>"][i]
        d.arrow(f"e1{i}", f"{s_}_box", f"{t_}_box", lbl, S.MUTED, (1, 0.5), (0, 0.5))

    d.arrow("e20", "a4_box", "b3_box", "p<sub>iv</sub> on a two-decimal grid",
            S.BLUE, (0.5, 1), (0.5, 0),
            points=[(XS[3] + BW // 2, GAP12 - 10), (XS[2] + BW // 2, GAP12 - 10)])
    # The coded reference sits two columns from the audit, so its edge runs along the empty
    # lane at the foot of the band rather than straight through the gold-set box.
    d.arrow("e21", "b1_box", "b3_box", "y<sub>iv</sub> coded", S.GREY, (0.5, 1), (0.2, 1),
            points=[(XS[0] + BW // 2, LANE2), (XS[2] + 65, LANE2)])
    d.arrow("e22", "b2_box", "b3_box", "y<sub>iv</sub> human", S.ORANGE, (1, 0.5), (0, 0.5))
    d.arrow("e23", "b2_box", "b4_box", "the same labels score every arm",
            S.ORANGE, (0.5, 0), (0.5, 0), dashed=True,
            points=[(XS[1] + BW // 2, 334), (XS[3] + BW // 2, 334)])
    d.arrow("e30", "b3_box", "c1_box", "risk-coverage over flagged positives",
            S.BLUE, (0.35, 1), (0.5, 0),
            points=[(XS[2] + 113, GAP23 - 14), (36 + CW // 2, GAP23 - 16)])
    d.arrow("e31", "b3_box", "c3_box", "calibrated p", S.BLUE, (0.8, 1), (0.5, 0),
            points=[(XS[2] + 259, GAP23 - 14), (984 + CW // 2, GAP23 - 16)])
    # The loop: the budget changes how the table is used, not what it holds. Routed out to
    # the right margin and back along the top lane so that it crosses no text.
    d.arrow("e32", "c3_box", "a4_box", "",
            S.MUTED, (1, 0.5), (0.5, 0), dashed=True,
            points=[(RIGHT, y3 + 88), (RIGHT, 8), (XS[3] + BW // 2, 8)])
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true",
                    help="also render PDF and PNG with the draw.io desktop CLI")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "F1_framework.drawio"
    f.write_text(build().xml(), encoding="utf-8")
    print(f"wrote {f}")

    if args.export:
        if not DRAWIO_EXE.exists():
            print(f"draw.io not found at {DRAWIO_EXE}; open the .drawio and export by hand")
            return
        for ext in ("pdf", "png"):
            cmd = [str(DRAWIO_EXE), "--export", "--format", ext, "--crop",
                   "--output", str(OUT / f"F1_framework.{ext}"), str(f)]
            if ext == "png":
                cmd[2:2] = ["--scale", "2"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            ok = (OUT / f"F1_framework.{ext}").exists()
            print(f"  {ext}: {'ok' if ok else 'FAILED'} {r.stderr.strip()[:120]}")


if __name__ == "__main__":
    main()
