#!/usr/bin/env python3
"""Generate the SVG diagrams used by docs/holy-approach.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "approach_assets"

NAVY = "#0d2b4e"
BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#b45309"
RED = "#b91c1c"
GREEN = "#15803d"
GRAY = "#64748b"
LIGHT_BLUE = "#eaf1fb"
LIGHT_TEAL = "#e7f6f3"
LIGHT_AMBER = "#fdf3e3"
LIGHT_RED = "#fdecec"
LIGHT_GRAY = "#f1f5f9"


def canvas(width=12, height=5.2):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body="",
    edge=BLUE,
    face=LIGHT_BLUE,
    title_size=10.5,
    body_size=8.5,
    title_color=NAVY,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color="#243447",
            linespacing=1.35,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=title_color,
            linespacing=1.3,
        )


def arrow(ax, start, end, color=GRAY, style="-|>", lw=1.8, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def diagram_thesis():
    fig, ax = canvas(12, 5.6)
    ax.text(50, 95, "One model or three stages?", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 88, "The assignment is decomposed because the constraints and the available supervision point in the same direction.", ha="center", fontsize=9.5, color="#334155")

    box(ax, 3, 11, 31, 66, "MONOLITHIC", "One large model\n\npunctuation +\nordering +\ntranslation\n\nrejected", edge=RED, face=LIGHT_RED, title_size=12)
    ax.text(18.5, 22, "no aligned training data\nopaque errors\ntoo large for 1.5 GB", ha="center", va="center", fontsize=8, color=RED, linespacing=1.25)

    box(ax, 36, 11, 61, 66, "", edge=NAVY, face="#ffffff")
    ax.text(66.5, 71, "DECOMPOSED PIPELINE", ha="center", fontsize=12, fontweight="bold", color=NAVY)
    box(ax, 39.5, 47, 17.5, 18, "STAGE 1", "mBERT + LoRA\n5 labels / word", edge=TEAL, face=LIGHT_TEAL, title_size=9.5, body_size=7.8)
    box(ax, 59.5, 47, 17.5, 18, "STAGE 2", "mBERT + LoRA\npair scorer + Borda", edge=BLUE, face=LIGHT_BLUE, title_size=9.5, body_size=7.8)
    box(ax, 79.5, 47, 15, 18, "STAGE 3", "IndicTrans2\nbeam search", edge=AMBER, face=LIGHT_AMBER, title_size=9.5, body_size=7.8)
    box(ax, 45, 18, 42, 17, "RESULT", "aligned Hindi/English sentences\n+ groups of five", edge=GREEN, face="#eaf7ee", title_size=9.5, body_size=8)
    arrow(ax, (57.2, 56), (59.3, 56), color=BLUE)
    arrow(ax, (77.2, 56), (79.3, 56), color=AMBER)
    arrow(ax, (68.2, 47), (68.2, 35.3), color=GREEN)
    arrow(ax, (48.2, 47), (60.5, 35.3), color=GREEN, dashed=True)
    ax.text(53, 40.5, "sentences", fontsize=7.5, color=GREEN, rotation=24)
    ax.text(43, 8.5, "each stage is trainable, measurable and replaceable", ha="center", fontsize=8.5, color="#334155")
    save(fig, "01_thesis.svg")


def diagram_pipeline():
    fig, ax = canvas(13, 4.6)
    ax.text(50, 92, "The end-to-end pipeline", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 85, "Every input file passes through the same three-stage contract.", ha="center", fontsize=9.5, color="#334155")

    boxes = [
        (2, 47, 16, 24, "INPUT", "raw Hindi\nwords + NULL", GRAY, LIGHT_GRAY),
        (21, 47, 16, 24, "CLEAN", "remove NULL\nkeep word order", GRAY, LIGHT_GRAY),
        (40, 47, 18, 24, "STAGE 1", "mBERT + LoRA\npunctuation labels", TEAL, LIGHT_TEAL),
        (61, 47, 18, 24, "STAGE 2", "pair scorer\n+ Borda decoder", BLUE, LIGHT_BLUE),
        (82, 47, 16, 24, "STAGE 3", "IndicTrans2\nbeam search", AMBER, LIGHT_AMBER),
    ]
    for x, y, w, h, title, body, edge, face in boxes:
        box(ax, x, y, w, h, title, body, edge=edge, face=face, title_size=10, body_size=8)
    for x1, x2 in [(18.2, 20.8), (37.2, 39.8), (58.2, 60.8), (79.2, 81.8)]:
        arrow(ax, (x1, 59), (x2, 59), color=GRAY)

    box(ax, 10, 10, 35, 23, "HINDI OUTPUT", "sentences_hi.txt\ngroups_hi.txt", edge=GREEN, face="#eaf7ee", title_size=10, body_size=8)
    box(ax, 55, 10, 35, 23, "ENGLISH OUTPUT", "sentences_en.txt\ngroups_en.txt", edge=GREEN, face="#eaf7ee", title_size=10, body_size=8)
    arrow(ax, (49, 47), (30, 33.5), color=GREEN)
    arrow(ax, (90, 47), (72, 33.5), color=GREEN)
    ax.text(50, 4, "Groups are consecutive blocks of five. Sentence i in Hindi maps to sentence i in English.", ha="center", fontsize=8.5, color="#334155")
    save(fig, "02_pipeline.svg")


def diagram_data():
    fig, ax = canvas(12, 6.2)
    ax.text(50, 95, "Where the supervision comes from", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 88, "Hindi Wikipedia already contains the punctuation and the true sentence order.", ha="center", fontsize=9.5, color="#334155")

    box(ax, 4, 66, 26, 15, "WIKIPEDIA", "wikimedia/wikipedia\n20231101.hi", edge=GRAY, face=LIGHT_GRAY)
    box(ax, 37, 66, 26, 15, "PARSE", "remove citations\nsplit sentences\nstrip punctuation", edge=TEAL, face=LIGHT_TEAL)
    box(ax, 70, 66, 26, 15, "ARTICLE SPLIT", "95% / 5% / 5%\n2700 / 150 / 150", edge=BLUE, face=LIGHT_BLUE)
    arrow(ax, (30.2, 73.5), (36.8, 73.5), color=GRAY)
    arrow(ax, (63.2, 73.5), (69.8, 73.5), color=GRAY)

    box(ax, 2, 26, 30, 23, "PUNCTUATION DATA", "32-96 word windows\nwhole sentences\nshuffled inside window\n5 labels per word", edge=TEAL, face=LIGHT_TEAL, title_size=9.5, body_size=8)
    box(ax, 35, 26, 30, 23, "ORDERING DATA", "consecutive pairs\n(A,B,1) + (B,A,0)\n40,000 train pairs", edge=BLUE, face=LIGHT_BLUE, title_size=9.5, body_size=8)
    box(ax, 68, 26, 30, 23, "ORDER EVALUATION", "held-out articles\n8-70 sentences\nshuffled at eval time\nKendall's tau", edge=AMBER, face=LIGHT_AMBER, title_size=9.5, body_size=8)
    arrow(ax, (83, 66), (17, 49.3), color=TEAL, dashed=True)
    arrow(ax, (83, 66), (50, 49.3), color=BLUE, dashed=True)
    arrow(ax, (83, 66), (83, 49.3), color=AMBER, dashed=True)
    ax.text(50, 15, "Same articles are used for both stages, so the split is comparable and no article leaks across train/val/test.", ha="center", fontsize=8.5, color="#334155")
    save(fig, "03_data.svg")


def diagram_punctuation():
    fig, ax = canvas(13, 5.4)
    ax.text(50, 94, "Stage 1: punctuation as word-level token classification", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 87, "The model never rewrites a word. It chooses one of five labels for each existing word.", ha="center", fontsize=9, color="#334155")

    box(ax, 2, 52, 13, 20, "WORDS", "WORD 1\nWORD 2\nWORD 3\nWORD 4\nWORD 5", edge=GRAY, face=LIGHT_GRAY, title_size=9, body_size=8)
    box(ax, 18, 52, 14, 20, "SUBWORDS", "first piece\n+ ## pieces\nword_ids()", edge=TEAL, face=LIGHT_TEAL, title_size=9, body_size=8)
    box(ax, 35, 52, 15, 20, "mBERT", "frozen backbone\n+ LoRA Q/V", edge=BLUE, face=LIGHT_BLUE, title_size=9, body_size=8)
    box(ax, 53, 52, 14, 20, "HEAD", "linear layer\n5 logits", edge=BLUE, face=LIGHT_BLUE, title_size=9, body_size=8)
    box(ax, 70, 52, 13, 20, "SOFTMAX", "P(label)\nper subword", edge=AMBER, face=LIGHT_AMBER, title_size=9, body_size=8)
    box(ax, 86, 52, 12, 20, "LABEL", "first\nsubword\nonly", edge=GREEN, face="#eaf7ee", title_size=9, body_size=8)
    for x1, x2 in [(15.2, 17.8), (32.2, 34.8), (50.2, 52.8), (67.2, 69.8), (83.2, 85.8)]:
        arrow(ax, (x1, 62), (x2, 62), color=GRAY)

    box(ax, 14, 13, 27, 24, "TRAINING TARGET", "none / comma /\nsentence_end /\nquestion / exclamation\n\ncontinuation subwords = -100", edge=TEAL, face=LIGHT_TEAL, title_size=9, body_size=7.8)
    box(ax, 45, 13, 27, 24, "LOSS", "class-weighted\ncross entropy\n\nrare marks get\nlarger weights", edge=AMBER, face=LIGHT_AMBER, title_size=9, body_size=7.8)
    box(ax, 76, 13, 22, 24, "RECONSTRUCTION", "insert predicted mark\nafter each word\n\nsplit at boundary\nlabels", edge=GREEN, face="#eaf7ee", title_size=9, body_size=7.8)
    arrow(ax, (41.2, 25), (44.8, 25), color=GRAY)
    arrow(ax, (72.2, 25), (75.8, 25), color=GRAY)
    ax.text(50, 5, "A word with several subword pieces is supervised once, at its first piece.", ha="center", fontsize=8.5, color="#334155")
    save(fig, "04_punctuation.svg")


def diagram_windows():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 94, "Overlapping windows and triangular weighting", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 87, "Long files are split into 64-word windows with a stride of 40; a word can be seen more than once.", ha="center", fontsize=9, color="#334155")

    for i, (start, end) in enumerate([(6, 43), (30, 67), (54, 91)]):
        y = 67 - i * 10
        ax.add_patch(Rectangle((start, y), end - start, 6, facecolor=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.2))
        ax.text(start + 1.2, y + 3, f"window {i + 1}", fontsize=8.5, color=NAVY, va="center")
    ax.text(96, 74, "64 words", fontsize=8, color=GRAY)
    ax.annotate("", xy=(96, 72), xytext=(96, 65), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1))

    x = np.linspace(0, 1, 200)
    weights = np.maximum(0.05, 1 - np.abs(x - 0.5) / 0.5)
    ax2 = fig.add_axes([0.14, 0.14, 0.72, 0.28])
    ax2.plot(x, weights, color=TEAL, linewidth=2.4)
    ax2.fill_between(x, weights, color=TEAL, alpha=0.18)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.08)
    ax2.set_xticks([0, 0.5, 1])
    ax2.set_xticklabels(["left edge", "center", "right edge"], fontsize=8)
    ax2.set_yticks([0.05, 0.5, 1.0])
    ax2.set_yticklabels(["0.05", "0.5", "1.0"], fontsize=8)
    ax2.set_ylabel("weight", fontsize=8)
    ax2.set_title("Weight applied to a word's probabilities inside a window", fontsize=9, color=NAVY)
    ax2.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    for spine in ax2.spines.values():
        spine.set_color("#cbd5e1")
    save(fig, "05_windows.svg")


def diagram_ordering():
    fig, ax = canvas(13, 6.2)
    ax.text(50, 95, "Stage 2: pairwise preferences, then a global decoder", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 88, "The model answers one simple question; the decoder turns many answers into one order.", ha="center", fontsize=9, color="#334155")

    box(ax, 2, 61, 19, 17, "SENTENCES", "shuffled\nHindi sentences", edge=GRAY, face=LIGHT_GRAY, title_size=9.5, body_size=8)
    box(ax, 25, 61, 21, 17, "PAIRS", "n(n-1)/2\nunordered pairs", edge=TEAL, face=LIGHT_TEAL, title_size=9.5, body_size=8)
    box(ax, 50, 61, 23, 17, "PAIR CLASSIFIER", "mBERT + LoRA\nP(A before B)", edge=BLUE, face=LIGHT_BLUE, title_size=9.5, body_size=8)
    box(ax, 77, 61, 21, 17, "SCORE MATRIX", "M[i,j] +\nM[j,i] = 1", edge=AMBER, face=LIGHT_AMBER, title_size=9.5, body_size=8)
    for x1, x2 in [(21.2, 24.8), (46.2, 49.8), (73.2, 76.8)]:
        arrow(ax, (x1, 69.5), (x2, 69.5), color=GRAY)

    box(ax, 5, 29, 25, 18, "BORDA", "average 'before'\nprobability\nsort descending", edge=GREEN, face="#eaf7ee", title_size=10, body_size=8)
    box(ax, 37, 29, 25, 18, "CLUSTERS", "TF-IDF + KMeans\norder medoids\norder inside", edge=BLUE, face=LIGHT_BLUE, title_size=10, body_size=8)
    box(ax, 69, 29, 26, 18, "BEAM", "keep 8 best\nchains\nappend best pair", edge=AMBER, face=LIGHT_AMBER, title_size=10, body_size=8)
    arrow(ax, (86, 60.8), (17.5, 47.2), color=GREEN, dashed=True)
    arrow(ax, (86, 60.8), (49.5, 47.2), color=BLUE, dashed=True)
    arrow(ax, (86, 60.8), (82, 47.2), color=AMBER, dashed=True)
    box(ax, 35, 6, 30, 14, "ONE PERMUTATION", "applied to the punctuated\nHindi sentences", edge=GREEN, face="#eaf7ee", title_size=9.5, body_size=8)
    arrow(ax, (17.5, 28.8), (35, 13), color=GREEN)
    arrow(ax, (49.5, 28.8), (50, 20.2), color=GREEN)
    arrow(ax, (82, 28.8), (65, 13), color=GREEN)
    save(fig, "06_ordering.svg")


def diagram_translation():
    fig, ax = canvas(12, 4.8)
    ax.text(50, 93, "Stage 3: sentence-aligned Hindi-to-English translation", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 86, "IndicTrans2 is an encoder-decoder; translation never changes the sentence count or the group structure.", ha="center", fontsize=9, color="#334155")

    boxes = [
        (2, 48, 16, 22, "ORDERED\nHINDI", "one sentence\nper item", GRAY, LIGHT_GRAY),
        (21, 48, 17, 22, "PROCESSOR", "hin_Deva /\neng_Latn\nnormalise", TEAL, LIGHT_TEAL),
        (41, 48, 17, 22, "ENCODER", "IndicTrans2\n200M", BLUE, LIGHT_BLUE),
        (61, 48, 16, 22, "DECODER", "beam search\n4 beams", AMBER, LIGHT_AMBER),
        (80, 48, 18, 22, "POSTPROCESS", "remove tags\nfix spacing", GREEN, "#eaf7ee"),
    ]
    for x, y, w, h, title, body, edge, face in boxes:
        box(ax, x, y, w, h, title, body, edge=edge, face=face, title_size=9.3, body_size=7.8)
    for x1, x2 in [(18.2, 20.8), (38.2, 40.8), (58.2, 60.8), (77.2, 79.8)]:
        arrow(ax, (x1, 59), (x2, 59), color=GRAY)

    box(ax, 27, 12, 46, 20, "ALIGNED ENGLISH", "same number of lines as the Hindi input\nempty input lines stay empty\nline i -> line i", edge=GREEN, face="#eaf7ee", title_size=10, body_size=8)
    arrow(ax, (89, 47.8), (73, 32.2), color=GREEN)
    ax.text(50, 5, "Batch size 4, max generation length 256, no fallback translator.", ha="center", fontsize=8.5, color="#334155")
    save(fig, "07_translation.svg")


def diagram_memory():
    fig, ax = canvas(12, 5.2)
    ax.text(50, 94, "The 1.5 GB strategy is architectural, not wishful", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 87, "Load exactly one model, run that stage on every file, free it, then load the next.", ha="center", fontsize=9, color="#334155")

    ax.plot([5, 95], [72, 72], color=RED, linewidth=2, linestyle="--")
    ax.text(96, 72, "1.5 GB cap", color=RED, fontsize=9, va="center")
    stages = [
        (8, 20, "STAGE 1", "mBERT punctuation", "~0.67 GB on MPS\nmeasured adapter run", TEAL, LIGHT_TEAL),
        (39, 21, "STAGE 2", "mBERT ordering", "~0.67 GB on MPS\nmeasured adapter run", BLUE, LIGHT_BLUE),
        (71, 20, "STAGE 3", "IndicTrans2 200M", "loaded alone\nthen freed", AMBER, LIGHT_AMBER),
    ]
    for x, w, title, body, note, edge, face in stages:
        box(ax, x, 38, w, 22, title, body, edge=edge, face=face, title_size=9.5, body_size=8)
        ax.text(x + w / 2, 31, note, ha="center", va="center", fontsize=7.5, color="#334155", linespacing=1.25)
    ax.annotate("", xy=(38, 49), xytext=(29, 49), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.8))
    ax.annotate("", xy=(70, 49), xytext=(61, 49), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.8))
    ax.text(33.5, 53.5, "free", ha="center", fontsize=8, color=GRAY)
    ax.text(65.5, 53.5, "free", ha="center", fontsize=8, color=GRAY)
    ax.text(50, 13, "Peak memory is bounded by the largest single stage, not by the sum of all three models.\nThe CUDA cap is enforced with torch.cuda.set_per_process_memory_fraction.", ha="center", fontsize=8.5, color="#334155", linespacing=1.4)
    save(fig, "08_memory.svg")


def diagram_errors():
    fig, ax = canvas(12, 5.2)
    ax.text(50, 94, "Error propagation: where quality is lost", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 87, "The file contract keeps outputs aligned, but each stage's mistakes change the next stage's input.", ha="center", fontsize=9, color="#334155")

    box(ax, 2, 55, 20, 19, "PUNCTUATION", "missed or extra\nboundary", edge=TEAL, face=LIGHT_TEAL, title_size=9.2, body_size=8)
    box(ax, 27, 55, 20, 19, "SEGMENTATION", "wrong sentence\ncount or length", edge=RED, face=LIGHT_RED, title_size=9.2, body_size=8)
    box(ax, 52, 55, 20, 19, "ORDERING", "wrong pair\npreferences", edge=BLUE, face=LIGHT_BLUE, title_size=9.2, body_size=8)
    box(ax, 77, 55, 20, 19, "TRANSLATION", "less coherent\ncontext", edge=AMBER, face=LIGHT_AMBER, title_size=9.2, body_size=8)
    for x1, x2 in [(22.2, 26.8), (47.2, 51.8), (72.2, 76.8)]:
        arrow(ax, (x1, 64.5), (x2, 64.5), color=RED)
    box(ax, 18, 18, 28, 20, "WHAT STILL HOLDS", "NULL removed\nHindi/English line i aligned\ngroup count = ceil(n/5)", edge=GREEN, face="#eaf7ee", title_size=9.3, body_size=7.8)
    box(ax, 55, 18, 28, 20, "WHAT DEGRADES", "readability\nfull-document order\nrare punctuation\ntranslation adequacy", edge=RED, face=LIGHT_RED, title_size=9.3, body_size=7.8)
    arrow(ax, (87, 54.8), (78, 38.2), color=RED, dashed=True)
    arrow(ax, (12, 54.8), (23, 38.2), color=GREEN, dashed=True)
    save(fig, "09_errors.svg")


def diagram_scores():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    labels = ["none", "comma", "sentence_end", "question", "exclamation"]
    values = [0.9285, 0.3161, 0.8838, 0.0, 0.0]
    colors = [GRAY, AMBER, TEAL, RED, RED]
    axes[0].barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("F1 on held-out punctuation windows", fontsize=9)
    axes[0].set_title("Punctuation: boundary F1 0.8842", fontsize=11, color=NAVY, loc="left")
    for i, value in enumerate(values[::-1]):
        axes[0].text(value + 0.02, i, f"{value:.4f}", va="center", fontsize=8, color="#243447")
    axes[0].grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axes[0].tick_params(labelsize=8)

    names = ["Borda", "clusters", "beam", "TF-IDF"]
    taus = [0.2007, 0.1220, 0.0921, 0.0631]
    colors = [GREEN, BLUE, AMBER, GRAY]
    axes[1].barh(names[::-1], taus[::-1], color=colors[::-1], height=0.55)
    axes[1].axvline(0, color="#94a3b8", linewidth=1)
    axes[1].set_xlim(0, 0.25)
    axes[1].set_xlabel("mean Kendall's tau over 40 articles", fontsize=9)
    axes[1].set_title("Ordering: pair accuracy 0.6543", fontsize=11, color=NAVY, loc="left")
    for i, value in enumerate(taus[::-1]):
        axes[1].text(value + 0.005, i, f"{value:.4f}", va="center", fontsize=8, color="#243447")
    axes[1].grid(axis="x", color="#e2e8f0", linewidth=0.8)
    axes[1].tick_params(labelsize=8)

    fig.suptitle("Measured local results: the honest scoreboard", fontsize=15, fontweight="bold", color=NAVY, x=0.06, ha="left")
    fig.text(0.06, 0.90, "These are subset-adapter results, not the full 25,000-window / 40,000-pair recipe.", fontsize=8.8, color="#334155")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.83])
    save(fig, "10_scores.svg")


def diagram_groups():
    fig, ax = canvas(12, 5.4)
    ax.text(50, 94, "Alignment contract: sentence i in, sentence i out", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 87, "Translating a flat list preserves line order; grouping is a pure chunking step.", ha="center", fontsize=9, color="#334155")

    ax.text(25, 78, "HINDI SENTENCES", ha="center", fontsize=10, fontweight="bold", color=TEAL)
    ax.text(75, 78, "ENGLISH SENTENCES", ha="center", fontsize=10, fontweight="bold", color=AMBER)
    for i in range(5):
        y = 68 - i * 9
        ax.add_patch(Rectangle((8, y - 3), 34, 6, facecolor=LIGHT_TEAL, edgecolor=TEAL, linewidth=1))
        ax.add_patch(Rectangle((58, y - 3), 34, 6, facecolor=LIGHT_AMBER, edgecolor=AMBER, linewidth=1))
        ax.text(10, y, f"H{i + 1}", fontsize=8.5, color=NAVY, va="center")
        ax.text(60, y, f"E{i + 1}", fontsize=8.5, color=NAVY, va="center")
        arrow(ax, (44, y), (56, y), color=GREEN, lw=1.2)
    ax.add_patch(Rectangle((5, 24), 40, 48, fill=False, edgecolor=GREEN, linewidth=2, linestyle="--"))
    ax.add_patch(Rectangle((55, 24), 40, 48, fill=False, edgecolor=GREEN, linewidth=2, linestyle="--"))
    ax.text(25, 18, "group 1 = sentences 1-5", ha="center", fontsize=8.5, color=GREEN)
    ax.text(75, 18, "group 1 = translations 1-5", ha="center", fontsize=8.5, color=GREEN)
    ax.text(50, 8, "No sorting, filtering, or re-alignment is allowed after translation.", ha="center", fontsize=8.5, color="#334155")
    save(fig, "11_groups.svg")


def diagram_decisions():
    fig, ax = canvas(12, 5.6)
    ax.text(50, 94, "The decision spine", ha="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(50, 87, "Each choice is tied to a constraint, a measurement, and an explicit alternative.", ha="center", fontsize=9, color="#334155")

    items = [
        ("LANGUAGE", "Hindi / Devanagari", "danda + mBERT", TEAL, LIGHT_TEAL),
        ("TOKENIZE", "whitespace split", "preserve matras", TEAL, LIGHT_TEAL),
        ("PUNCTUATION", "5-label classifier", "preserve every word", BLUE, LIGHT_BLUE),
        ("BACKBONE", "mBERT + LoRA", "fit the budget", BLUE, LIGHT_BLUE),
        ("ORDERING", "pairwise + decoder", "any document length", AMBER, LIGHT_AMBER),
        ("TRANSLATE", "IndicTrans2 200M", "Indic-specialized", AMBER, LIGHT_AMBER),
        ("MEMORY", "load / free stages", "peak = one model", GREEN, "#eaf7ee"),
        ("EVALUATE", "per-stage metrics", "localize failures", GREEN, "#eaf7ee"),
    ]
    for index, (title, body, reason, edge, face) in enumerate(items):
        row, column = divmod(index, 4)
        x = 4 + column * 23.5
        y = 57 - row * 31
        box(ax, x, y, 21, 21, title, body + "\n\n" + reason, edge=edge, face=face, title_size=8.8, body_size=7.4)
    ax.text(50, 5, "Every row has a rejected alternative and a measured consequence in the full decision log.", ha="center", fontsize=8.5, color="#334155")
    save(fig, "12_decisions.svg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    diagram_thesis()
    diagram_pipeline()
    diagram_data()
    diagram_punctuation()
    diagram_windows()
    diagram_ordering()
    diagram_translation()
    diagram_memory()
    diagram_errors()
    diagram_scores()
    diagram_groups()
    diagram_decisions()


if __name__ == "__main__":
    main()
