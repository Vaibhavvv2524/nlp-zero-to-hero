#!/usr/bin/env python3
"""Generate SVG diagrams for the priority-order learning guide."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "learning_assets"

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
LIGHT_GREEN = "#eaf7ee"
LIGHT_GRAY = "#f1f5f9"


def canvas(width=12, height=5.4):
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
    body_size=8.2,
    title_color=NAVY,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.32,rounding_size=1.0",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.65,
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


def arrow(ax, start, end, color=GRAY, style="-|>", lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def diagram_priority_ladder():
    fig, ax = canvas(12, 8.2)
    ax.text(50, 96, "Study in this order: P0 to P7", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 90, "Each level earns the next. If time runs out, stop after P3 and still be able to explain the project.", ha="center", fontsize=9, color="#334155")

    rows = [
        ("P0", "THE STORY", "Three stages, no LLM, one model per stage, exact alignment", "Say it in 60 seconds.", GREEN, LIGHT_GREEN),
        ("P1", "STAGE 1", "Punctuation as 5-label word classification; subwords; imbalance", "Explain accuracy vs boundary F1.", TEAL, LIGHT_TEAL),
        ("P2", "STAGE 2", "Pairwise A-before-B; reversals; score matrix; Borda", "Explain why tau is only 0.20.", AMBER, LIGHT_AMBER),
        ("P3", "STAGE 3", "IndicTrans2; sentence-level translation; BLEU and chrF", "Explain alignment and drift.", BLUE, LIGHT_BLUE),
        ("P4", "DATA + METRICS", "Hide punctuation; consecutive pairs; article splits", "Know the scoreboard and why it is honest.", TEAL, LIGHT_TEAL),
        ("P5", "SYSTEMS", "LoRA; sequential loading; 1.5GB cap; no fallback", "Explain how memory stays bounded.", BLUE, LIGHT_BLUE),
        ("P6", "LIMITS", "Weak ordering; weak commas; dialogue; rejected alternatives", "State what you must not claim.", RED, LIGHT_RED),
        ("P7", "CODE + DEMO", "Pipeline map; demo notebook; run and inspect outputs", "Trace one sentence end to end.", GRAY, LIGHT_GRAY),
    ]

    top = 83
    height = 8.0
    gap = 1.3
    for index, (tag, title, body, check, edge, face) in enumerate(rows):
        y = top - index * (height + gap)
        ax.add_patch(
            FancyBboxPatch(
                (3, y),
                94,
                height,
                boxstyle="round,pad=0.25,rounding_size=0.9",
                linewidth=1.0,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.add_patch(Rectangle((4.2, y + 1.0), 8.2, height - 2.0, facecolor=edge, edgecolor="none"))
        ax.text(8.3, y + height / 2, tag, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(15, y + height * 0.66, title, ha="left", va="center", fontsize=10, fontweight="bold", color=NAVY)
        ax.text(15, y + height * 0.30, body, ha="left", va="center", fontsize=7.8, color="#243447")
        ax.text(96, y + height / 2, check, ha="right", va="center", fontsize=7.6, color=edge, style="italic")

    ax.text(50, 1.5, "Priority is not the same as page count: P0 and P1 carry the most marks per minute.", ha="center", fontsize=8, color=GRAY)
    save(fig, "01_priority_ladder.svg")


def diagram_pipeline_story():
    fig, ax = canvas(12, 5.6)
    ax.text(50, 94, "The 60-second story", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 88, "Unpunctuated, shuffled Hindi words in; ordered Hindi and English out.", ha="center", fontsize=9, color="#334155")

    box(ax, 2, 44, 15, 25, "INPUT", "Hindi words\nno punctuation\nshuffled sentences\nNULL removed", edge=GRAY, face=LIGHT_GRAY, title_size=10)
    box(ax, 21, 40, 17, 33, "STAGE 1", "Punctuation\n+ boundaries\n\nmBERT + LoRA\n5 labels per word", edge=TEAL, face=LIGHT_TEAL, title_size=11)
    box(ax, 42, 40, 17, 33, "STAGE 2", "Sentence order\n\npairwise mBERT\nA before B?\nBorda decode", edge=AMBER, face=LIGHT_AMBER, title_size=11)
    box(ax, 63, 40, 17, 33, "STAGE 3", "Translate\n\nIndicTrans2 200M\nsentence by sentence\nbeam search", edge=BLUE, face=LIGHT_BLUE, title_size=11)
    box(ax, 84, 44, 14, 25, "OUTPUT", "Hindi sentences\nEnglish sentences\ngroups of five\naligned indexes", edge=GREEN, face=LIGHT_GREEN, title_size=10)

    arrow(ax, (17.5, 56), (20.5, 56), color=GRAY)
    arrow(ax, (38.5, 56), (41.5, 56), color=GRAY)
    arrow(ax, (59.5, 56), (62.5, 56), color=GRAY)
    arrow(ax, (80.5, 56), (83.5, 56), color=GRAY)

    ax.text(29.5, 32, "words -> sentences", ha="center", fontsize=7.8, color=TEAL)
    ax.text(50.5, 32, "sentences -> permutation", ha="center", fontsize=7.8, color=AMBER)
    ax.text(71.5, 32, "sentences -> English", ha="center", fontsize=7.8, color=BLUE)

    ax.add_patch(Rectangle((2, 11), 96, 12, facecolor="#f8fafc", edgecolor="#dbe3ee", linewidth=1))
    ax.text(50, 20, "INVARIANTS", ha="center", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.text(50, 14.5, "preserve words  |  permute only whole sentences  |  one output per input  |  load one model at a time", ha="center", fontsize=8, color="#334155")

    ax.text(50, 4, "No LLM. Only mBERT (stages 1-2) and IndicTrans2 (stage 3).", ha="center", fontsize=8, color=GRAY)
    save(fig, "02_pipeline_story.svg")


def diagram_time_plan():
    fig, ax = canvas(12, 5.9)
    ax.text(50, 94, "Three study plans", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 88, "Pick the plan that matches the time you actually have.", ha="center", fontsize=9, color="#334155")

    plans = [
        ("30 MINUTES", "SURVIVAL", ["Read P0 and P1", "Memorize 6 headline scores", "Read the 5 limitations", "Say the 60-second story"], GREEN, LIGHT_GREEN),
        ("2 HOURS", "CONFIDENT", ["P0 + P1 + P2 + P3", "Skim P4 data/metrics", "Open the executed demo", "Practice 10 self-test questions"], AMBER, LIGHT_AMBER),
        ("1 DAY", "DEFENSIBLE", ["All priorities P0-P7", "Trace one sentence in code", "Run one sample or read logs", "Read holy-approach 13-15", "Practice all 15 questions"], BLUE, LIGHT_BLUE),
    ]
    for index, (title, subtitle, items, edge, face) in enumerate(plans):
        x = 3 + index * 32.5
        ax.add_patch(
            FancyBboxPatch(
                (x, 13),
                29,
                67,
                boxstyle="round,pad=0.35,rounding_size=1.2",
                linewidth=1.2,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(x + 14.5, 73, title, ha="center", fontsize=11, fontweight="bold", color=NAVY)
        ax.text(x + 14.5, 67, subtitle, ha="center", fontsize=8, color=edge, fontweight="bold")
        for row, item in enumerate(items):
            y = 56 - row * 9.5
            ax.text(x + 3, y, "•", ha="left", fontsize=10, color=edge)
            ax.text(x + 6, y, item, ha="left", va="center", fontsize=7.8, color="#243447")

    ax.text(50, 4, "Stop rule: if you can explain P0-P3, the metrics, and the limitations, you are ready for a short presentation.", ha="center", fontsize=8, color=GRAY)
    save(fig, "03_time_plan.svg")


def diagram_metrics_snapshot():
    fig, ax = canvas(12, 5.8)
    ax.text(50, 95, "The honest scoreboard", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 89, "Strong sentence endings, weak commas and rare labels, modest ordering, usable translation.", ha="center", fontsize=9, color="#334155")

    box(ax, 3, 48, 29, 31, "STRONG", "boundary F1 0.884\nsentence_end F1 0.884\nnone F1 0.929\n\nsentence endings are reliable", edge=GREEN, face=LIGHT_GREEN, title_size=11)
    box(ax, 35.5, 48, 29, 31, "MODEST", "comma F1 0.316\npair accuracy 0.654\nBorda tau 0.201\n\nuseful but not solved", edge=AMBER, face=LIGHT_AMBER, title_size=11)
    box(ax, 68, 48, 29, 31, "WEAK / ZERO", "question F1 0.000\nexclamation F1 0.000\nclusters tau 0.122\nbeam tau 0.092", edge=RED, face=LIGHT_RED, title_size=11)

    box(ax, 3, 16, 45, 24, "TRANSLATION", "BLEU 34.125\nchrF 59.686\n\n100 IITB pairs, not assignment outputs", edge=BLUE, face=LIGHT_BLUE, title_size=11)
    box(ax, 52, 16, 45, 24, "BASELINE", "TF-IDF tau 0.063\n\nBorda beats the lexical baseline by about 3x", edge=TEAL, face=LIGHT_TEAL, title_size=11)

    ax.text(50, 8, "All numbers are local subset-adapter results, not the full 25k/40k recipe.", ha="center", fontsize=8, color=GRAY)
    save(fig, "04_metrics_snapshot.svg")


def diagram_abbreviations():
    fig, ax = canvas(12, 7.0)
    ax.text(50, 96, "Abbreviation map", ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(50, 90, "You do not need to memorize all of these. Recognize the ones marked with a star.", ha="center", fontsize=9, color="#334155")

    groups = [
        ("MODELS", [("*BERT", "Bidirectional Encoder\nRepresentations from Transformers"), ("*mBERT", "multilingual BERT"), ("*LLM", "Large Language Model"), ("MT", "Machine Translation"), ("IT2", "IndicTrans2")], TEAL, LIGHT_TEAL),
        ("TRAINING", [("*LoRA", "Low-Rank Adaptation"), ("PEFT", "Parameter-Efficient\nFine-Tuning"), ("FP16", "16-bit floating point"), ("FP32", "32-bit floating point"), ("CE", "Cross Entropy")], AMBER, LIGHT_AMBER),
        ("METRICS", [("*F1", "Precision-recall balance"), ("*BLEU", "n-gram translation score"), ("*chrF", "character n-gram F-score"), ("tau", "Kendall rank correlation"), ("acc", "accuracy")], BLUE, LIGHT_BLUE),
        ("SYSTEMS", [("*GPU", "Graphics Processing Unit"), ("*CUDA", "NVIDIA GPU runtime"), ("*MPS", "Apple Metal backend"), ("SDPA", "scaled dot-product attention"), ("OOM", "out of memory")], GREEN, LIGHT_GREEN),
        ("DATA", [("*NLP", "Natural Language Processing"), ("TF-IDF", "term frequency-inverse\ndocument frequency"), ("JSONL", "one JSON object per line"), ("API", "application programming\ninterface"), ("TOC", "table of contents")], GRAY, LIGHT_GRAY),
    ]

    for index, (title, items, edge, face) in enumerate(groups):
        x = 2.5 + index * 19.3
        ax.add_patch(
            FancyBboxPatch(
                (x, 10),
                18,
                72,
                boxstyle="round,pad=0.25,rounding_size=0.9",
                linewidth=1.0,
                edgecolor=edge,
                facecolor=face,
            )
        )
        ax.text(x + 9, 77, title, ha="center", fontsize=8.6, fontweight="bold", color=NAVY)
        for row, (abbr, full) in enumerate(items):
            y = 66 - row * 13
            ax.text(x + 9, y, abbr, ha="center", va="center", fontsize=8.2, fontweight="bold", color=edge)
            ax.text(x + 9, y - 4.2, full, ha="center", va="center", fontsize=6.4, color="#243447", linespacing=1.1)

    ax.text(50, 3, "The full table in Section 11 gives every full form and why it matters.", ha="center", fontsize=8, color=GRAY)
    save(fig, "05_abbreviations.svg")


def main():
    diagram_priority_ladder()
    diagram_pipeline_story()
    diagram_time_plan()
    diagram_metrics_snapshot()
    diagram_abbreviations()


if __name__ == "__main__":
    main()
