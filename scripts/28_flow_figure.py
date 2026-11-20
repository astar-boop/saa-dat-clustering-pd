"""
28 - Flow diagram of the sample selection.

Reconciles in a single scheme all the counts cited in the text, distinguishing
at each step the sample level from the participant level.

The counts are NOT hard-coded: they are recomputed here from the same sources
as scripts 01, 06 and 08, so that the diagram cannot diverge from the data.
With ONLY_SPORADIC=1 an extra step appears for the exclusion of the genetic
cohort, in both protocols and in the positive control.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _paths import DATA, curated_xlsx

BASE = os.environ.get("CLUSTERING_BASE",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(BASE, "figures")
ONLY = os.environ.get('ONLY_SPORADIC', "1") == "1"

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": int(os.environ.get("FIG_DPI", "200")), "font.size": 8.5,
                     "hatch.linewidth": 0.55})
BLUE, RED, GREY = "#1f5f8b", "#c44536", "#5c6f7b"
PANEL = "#eef2f5"
PINK = "#f9eeec"

# Colour is not the only channel: the fills 240 versus 241 in greyscale
# distinguished nothing. Each TYPE of box now has its own line style and
# hatching, legible in black and white too.
HATCHING = (0, (5, 3))
STYLES = {
    # ordinary step of the selection
    'stream': dict(ls="solid", lw=1.4, hatch=None, fc="white", ec=BLUE),
    # restriction / step off the main path
    'restriction': dict(ls=HATCHING, lw=1.6, hatch="/////", fc=PINK, ec=RED),
    # cohort analysed (end point of the main path)
    'cohort': dict(ls="solid", lw=2.8, hatch=None, fc=PANEL, ec=BLUE),
    # end point of the verification branch: hatching slanted the opposite way,
    # so the two red types stay distinct from each other as well
    'control': dict(ls=HATCHING, lw=2.6, hatch="\\\\\\\\\\", fc=PINK, ec=RED),
    # summary box, outside the cascade
    'remark': dict(ls="solid", lw=1.4, hatch=None, fc="white", ec=GREY),
}

# ===========================================================================
# counts, recomputed from the sources
# ===========================================================================
saa = pd.read_csv(os.path.join(DATA, "SAA_Biospecimen_Analysis_Results_24Apr2026.csv"))
cur = pd.read_excel(curated_xlsx(BASE),
                    sheet_name="20260511")
cbl = cur[cur.EVENT_ID == "BL"]
sg = cbl.set_index("PATNO").subgroup
sporadic = set(sg[sg == "Sporadic PD"].index)

# --- 24-hour protocol ---
s24 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 237) & (saa.CLINICAL_EVENT == "BL")]
t1 = s24[(s24.SAA_Status == "Positive") & (s24.SAA_Type == "Type1")]
n_bl24, s_bl24 = len(s24), s24.PATNO.nunique()
n_t1, s_t1 = len(t1), t1.PATNO.nunique()
s_cur24 = len(set(t1.PATNO) & set(cbl.PATNO))
s_spor24 = len(set(t1.PATNO) & set(cbl.PATNO) & sporadic)
n_prim = len(pd.read_csv(os.path.join(OUT, "analysis_complete_cases.csv")))

# --- 150-hour protocol ---
s150 = saa[(saa.COHORT == "PD") & (saa.PROJECTID == 155) & (saa.CLINICAL_EVENT == "BL")]
pos150 = s150[s150.SAA_Status == "Positive"]
n_bl150, s_bl150 = len(s150), s150.PATNO.nunique()
s_pos150 = pos150.PATNO.nunique()
s_spor150 = len(set(pos150.PATNO) & sporadic)
d150 = pd.read_csv(os.path.join(OUT, "08_dataset_150h.csv"))
d150 = d150[d150.complete] if "complete" in d150.columns else d150
n_rep = len(d150)
n_ov = len(set(d150.PATNO) & set(pd.read_csv(
    os.path.join(OUT, "analysis_complete_cases.csv")).PATNO))

# --- positive control (as in 06_sensitivity.py, section D) ---
ext = s24.copy()
for f in ["Fmax", "AUC", "SLOPEMax"]:
    cols = [f"{f}_24h_Rep{i}" for i in (1, 2, 3)]
    ext[f"saa_{f}"] = ext[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
ext = ext.groupby("PATNO", as_index=False).agg(
    {"saa_Fmax": "mean", "saa_AUC": "mean", "saa_SLOPEMax": "mean",
     "SAA_Status": "first"})
ext = ext.merge(cbl[["PATNO", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R",
                     "MIA_CAUDATE_BILAT", "age_at_visit"]], on="PATNO", how="inner")
ext["dat_putamen_low"] = ext[["MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]].min(axis=1)
if ONLY:
    ext = ext[ext.PATNO.isin(sporadic)]
ext = ext.dropna(subset=["saa_Fmax", "saa_AUC", "saa_SLOPEMax",
                         "dat_putamen_low", "MIA_CAUDATE_BILAT", "age_at_visit"])
vc = ext.SAA_Status.value_counts()
n_ext = len(ext)
n_ext_pos = int(vc.get("Positive", 0))
n_ext_neg = int(vc.get("Negative", 0))
n_ext_inc = int(vc.get("Inconclusive", 0))

# ===========================================================================
# drawing
# ===========================================================================
fig, ax = plt.subplots(figsize=(16.4, 7.8))
# the negative margin on the left hosts the verification branch, which used to
# pass inside the boxes of the column
ax.set_xlim(-9, 142); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, kind='stream', fs=8.5, bold=False):
    st = STYLES[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=st["fc"], ec=st["ec"], lw=st["lw"],
                                ls=st["ls"], hatch=st["hatch"], zorder=2))
    # on the hatched boxes the text goes on a solid background, otherwise the
    # hatching lines cross its glyphs
    bbox = (dict(boxstyle="square,pad=0.25", fc=st["fc"], ec="none")
            if st["hatch"] else None)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.5,
            bbox=bbox, zorder=3)


def arrow(x1, y1, x2, y2, colour=GREY, ls="solid", lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=colour, lw=lw,
                                 linestyle=ls, zorder=4))


def excluded(x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=7.6, color=RED,
            style="italic", linespacing=1.4)


def column(x, title, steps, y_high=92, y_lower=44, w=40, x_note=None):
    """Stacks the steps and branches the exclusions off the transition arrow."""
    ax.text(x + w / 2, y_high + 4, title, ha="center", fontsize=10,
            fontweight="bold", color=BLUE)
    n = len(steps)
    step = (y_high - y_lower) / max(n - 1, 1)
    if x_note is None:
        x_note = x + w + 3
    for i, p in enumerate(steps):
        h = p.get("h", 7.5)
        y = y_high - i * step - h
        box(x, y, w, h, p['text'], kind=p.get('kind', 'stream'),
            bold=p.get("bold", False))
        if i < n - 1:
            # The exclusion belongs to the transition that leads to the next
            # step: the horizontal branch avoids making it look like a
            # feature of the destination box.
            y_end = y - (step - h) + 0.5
            arrow(x + w / 2, y, x + w / 2, y_end)
            subsequent = steps[i + 1]
            if subsequent.get('excluded'):
                y_branch = (y + y_end) / 2
                arrow(x + w / 2, y_branch, x_note - 1.0, y_branch,
                        colour=RED, lw=1.2)
                excluded(x_note, y_branch, subsequent['excluded'])
    return y_high - (n - 1) * step - steps[-1].get("h", 7.5)


# ---- left column: 24 hours ----
steps24 = [
    dict(text=f"PD with an assay at the baseline visit\n{n_bl24} samples · {s_bl24} subjects",
         bold=True, h=7),
    dict(text=f"Type 1 positives (LBD-like)\n{n_t1} samples · {s_t1} subjects",
         excluded=f"{s_bl24 - s_t1} subjects excluded:\nnegative, inconclusive,\ntype 2, undetermined"),
    dict(text=f"Linked to the curated data cut\n{s_cur24} subjects",
         excluded=f"{s_t1 - s_cur24} subjects absent from the\nbaseline visit of the data cut"),
]
if ONLY:
    steps24.append(dict(
        text=f"Genetic cohort excluded\n{s_spor24} putatively sporadic PD",
        kind='restriction',
        excluded=f"{s_cur24 - s_spor24} carriers excluded:\n"
                "LRRK2, GBA1, SNCA, PRKN or\ncombinations"))
steps24.append(dict(
    text=f"PRIMARY COHORT\ncomplete kinetics, DAT and age\n{n_prim} subjects",
    kind='cohort', bold=True, h=8.5,
    excluded=f'{(s_spor24 if ONLY else s_cur24) - n_prim} subjects without\nDAT-SPECT or age'))
y_end24 = column(4, "Amprion 24 h protocol (Project 237)", steps24,
                   x_note=47)

# ---- positive control branch ----
y_ctrl = 24
box(4, y_ctrl, 40, 8.5,
    "EXTENDED POPULATION (positive control)\n"
    f"{n_ext} subjects: {n_ext_pos} positive,\n{n_ext_neg} negative, {n_ext_inc} inconclusive",
    kind='control', bold=True)

# The branch leaves from the side of the first box, runs down IN THE MARGIN on
# the left and re-enters the control box: dashed (non-chromatic channel) and
# labelled in the figure, so it is recognised without colour and without caption.
X_BRANCH = -4.0
y_top_mid = 92 - 7 / 2          # middle of the first box of the column
y_ctrl_mid = y_ctrl + 8.5 / 2
EDGE = 4 - 0.6                 # left edge of the boxes, padding included
ax.plot([EDGE, X_BRANCH, X_BRANCH, EDGE - 1.6],
        [y_top_mid, y_top_mid, y_ctrl_mid, y_ctrl_mid],
        color=RED, lw=1.8, ls=HATCHING, solid_capstyle="butt", zorder=4)
arrow(EDGE - 1.6, y_ctrl_mid, EDGE - 0.05, y_ctrl_mid,
        colour=RED, lw=1.8)
ax.text(X_BRANCH, (y_top_mid + y_ctrl_mid) / 2,
        "bypass: pipeline check (§3.6)", rotation=90, ha="center",
        va="center", fontsize=8, color=RED, fontweight="bold", zorder=5,
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none"))

excluded(46, y_ctrl + 4, "of the assayed subjects:\nno DAT-SPECT or age"
        + (",\ncarriers excluded" if ONLY else ""))
ax.text(24, y_ctrl - 3,
        "Both counts are at subject level.\n"
        f"The {n_ext} comprise the {n_prim} of the primary cohort plus the\n"
        "non-type-1 positives, negatives and inconclusives.",
        ha="center", va="top", fontsize=7.6, color=GREY, style="italic",
        linespacing=1.5)

# ---- right column: 150 hours ----
steps150 = [
    dict(text=f"PD with an assay at the baseline visit\n{n_bl150} samples · {s_bl150} subjects",
         bold=True, h=7),
    dict(text=f"Positive\n{s_pos150} subjects",
         excluded=f"{s_bl150 - s_pos150} subjects excluded:\nnegative and inconclusive\n"
                 "(type 1 / type 2 subtyping\nunavailable in Project 155)"),
]
if ONLY:
    steps150.append(dict(
        text=f"Genetic cohort excluded\n{s_spor150} putatively sporadic PD",
        kind='restriction',
        excluded=f"{s_pos150 - s_spor150} excluded:\n"
                f"{s_pos150 - s_spor150 - 1} carriers plus 1 unclassified\n"
                "(the 150 h cohort is far more\ngenetically enriched)"))
steps150.append(dict(
    text=f"REPLICATION COHORT\ncomplete kinetics, DAT and age\n{n_rep} subjects",
    kind='cohort', bold=True, h=8.5,
    excluded=f'{(s_spor150 if ONLY else s_pos150) - n_rep} subjects without\nDAT-SPECT or age'))
y_end150 = column(80, "150 h pilot protocol (Project 155)", steps150,
                    x_note=123)

# the overlap box must be kept below the last step of the
# right column, otherwise the two overlap each other
y_ov = min(y_ctrl - 4, y_end150 - 12)
box(80, y_ov, 40, 8,
    f"Overlap between the two cohorts\n{n_ov} subjects ({100*n_ov/n_rep:.0f}% of the replication cohort)",
    kind='remark')
arrow(100, y_end150, 100, y_ov + 8)
arrow(44, y_end24 + 2, 80, y_ov + 4, colour=GREY)

# ---- key to the encodings (line style and hatching, not colour) ----
# placed in the free space between the two columns: above are the exclusion
# notes (y ~26), below the closing note (y ~7)
LEG_X, LEG_Y, LEG_DY = 48.0, 8.2, 2.7
items = [
    ('stream', "Selection step"),
    ('restriction', "Restriction or off-path step"),
    ('cohort', "Analysed cohort"),
    ('control', "Endpoint of the bypass"),
]
for i, (kind, label) in enumerate(items):
    st = STYLES[kind]
    y = LEG_Y + 4 * LEG_DY - i * LEG_DY
    ax.add_patch(FancyBboxPatch((LEG_X + 1.2, y), 5.0, 1.6,
                                boxstyle="round,pad=0.35", fc=st["fc"],
                                ec=st["ec"], lw=st["lw"], ls=st["ls"],
                                hatch=st["hatch"], zorder=2))
    ax.text(LEG_X + 8.6, y + 0.8, label, ha="left", va="center",
            fontsize=7.8, color=GREY)
y_branch_leg = LEG_Y + 0.8
ax.plot([LEG_X + 1.5, LEG_X + 5.2], [y_branch_leg] * 2, color=RED, lw=1.8,
        ls=HATCHING, solid_capstyle="butt", zorder=4)
arrow(LEG_X + 5.2, y_branch_leg, LEG_X + 6.6, y_branch_leg, colour=RED, lw=1.8)
ax.text(LEG_X + 8.6, y_branch_leg, "Bypass to the pipeline check (§3.6)",
        ha="left", va="center", fontsize=7.8, color=GREY)
ax.text(LEG_X + 0.6, LEG_Y + 4 * LEG_DY + 3.2, "Encoding", ha="left",
        va="center", fontsize=8.4, fontweight="bold", color=BLUE)

ax.text(71, 5, "Exclusions branch to the right from the corresponding transition. "
        "Sample-level and subject-level counts diverge only in the\nuppermost "
        "boxes, because of repeat draws at the same visit.",
        ha="center", va="center", fontsize=8, color=GREY, linespacing=1.6)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig16_flow.png"), bbox_inches="tight")
print(f"saved fig16_flow.png  (ONLY_SPORADIC={('1' if ONLY else '0')})")
print(f"  24h: {n_bl24}/{s_bl24} -> {n_t1}/{s_t1} -> {s_cur24}"
      + (f" -> {s_spor24}" if ONLY else "") + f" -> {n_prim}")
print(f"  150h: {n_bl150}/{s_bl150} -> {s_pos150}"
      + (f" -> {s_spor150}" if ONLY else "") + f" -> {n_rep}")
print(f'  positive control: {n_ext} ({n_ext_pos}/{n_ext_neg}/{n_ext_inc}); overlap {n_ov}')
