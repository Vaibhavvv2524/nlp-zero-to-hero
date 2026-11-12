#!/usr/bin/env python3
"""Build the study-guide HTML and PDF from docs/part*.md."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PARTS = ["part1.md", "part2.md", "part3.md", "part4.md", "part5.md"]

COVER = """
<section class="cover">
  <div class="kicker">Study guide</div>
  <h1>NLP Zero to Hero</h1>
  <p class="subtitle">Hindi Text Restoration and Translation</p>
  <p class="lede">From first principles to a working three-stage pipeline: punctuation,
  sentence ordering, and translation. Every model, every source file, every notebook cell,
  and every measured result, explained.</p>
  <div class="meta">
    <p><strong>Pipeline:</strong> mBERT + LoRA &rarr; pairwise ordering &rarr; IndicTrans2</p>
    <p><strong>Constraints:</strong> no LLM &middot; 1.5 GB GPU memory &middot; three models, loaded one at a time</p>
    <p><strong>Repository:</strong> github.com/Vaibhavvv2524/nlp-zero-to-hero</p>
    <p><strong>Verified locally:</strong> macOS M4 (MPS), subset adapters, executed demo notebook</p>
  </div>
</section>
"""

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Charter", "Georgia", "Devanagari Sangam MN", "Noto Serif Devanagari", serif;
  font-size: 10.4pt; line-height: 1.55; color: #1b1f24; margin: 0;
}
h1, h2, h3, h4 { font-family: "Avenir Next", "Helvetica Neue", "Devanagari Sangam MN", sans-serif;
  color: #12355b; page-break-after: avoid; break-after: avoid; }
h1 { font-size: 20pt; margin: 0 0 8pt; padding-bottom: 4pt; border-bottom: 2px solid #d7e3f4; }
h2 { font-size: 13.6pt; margin: 18pt 0 6pt; }
h3 { font-size: 11.6pt; margin: 14pt 0 4pt; }
h4 { font-size: 10.6pt; margin: 12pt 0 4pt; }
p { margin: 0 0 7pt; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 8pt; padding-left: 16pt; }
li { margin: 0 0 3pt; }
a { color: #12355b; text-decoration: none; }
strong { color: #0f2a48; }
hr { border: 0; border-top: 1px solid #e3e8ef; margin: 14pt 0; }
blockquote {
  margin: 10pt 0; padding: 7pt 10pt; background: #f5f7fa;
  border-left: 3px solid #8aa4c8; color: #2b3644; page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
code {
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 8.7pt; background: #f2f4f7; padding: 0.5pt 3pt; border-radius: 2pt;
  color: #1f3550;
}
pre { margin: 8pt 0 10pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.45; }
.codehilite {
  background: #f6f8fa; border: 1px solid #e3e8ef; border-radius: 3pt;
  padding: 7pt 9pt; overflow: hidden; page-break-inside: avoid;
}
table {
  width: 100%; border-collapse: collapse; margin: 8pt 0 12pt;
  font-size: 8.9pt; page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d8dee7; padding: 3.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #eaf0f8; color: #12355b; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafbfd; }
.approach, .principles, .decision, .callout {
  margin: 10pt 0 12pt; padding: 8pt 11pt; border-radius: 3pt;
  page-break-inside: avoid; font-size: 9.7pt; line-height: 1.5;
}
.approach {
  background: #eef4ff; border-left: 4px solid #2563eb; color: #14294a;
}
.principles {
  background: #f2fbf5; border-left: 4px solid #1f8a4c; color: #143522;
}
.decision {
  background: #fff7ea; border-left: 4px solid #c47b12; color: #3d2a08;
}
.callout {
  background: #f7f2ff; border-left: 4px solid #6d4bb5; color: #2c1d4a;
}
.approach p:last-child, .principles p:last-child,
.decision p:last-child, .callout p:last-child { margin-bottom: 0; }
.pagebreak { display: block; height: 0; margin: 0; padding: 0;
  page-break-after: always; break-after: page; }
.toc-title { margin-bottom: 10pt; }
.toc { font-size: 9.3pt; column-count: 2; column-gap: 9mm;
  page-break-after: always; break-after: page; }
.toc > ul { list-style: none; padding-left: 0; margin: 0; }
.toc ul { list-style: none; padding-left: 9pt; margin: 2pt 0; }
.toc li { margin: 0 0 2.5pt; break-inside: avoid; }
.toc > ul > li { font-weight: 600; color: #12355b; margin-top: 5pt; }
.toc a { color: inherit; }
.cover { padding-top: 26mm; page-break-after: always; break-after: page; }
.cover .kicker {
  font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 9pt;
  letter-spacing: 2.2pt; text-transform: uppercase; color: #5b6b80; margin-bottom: 10pt;
}
.cover h1 { font-size: 40pt; line-height: 1.05; border: none; margin: 0 0 8pt; color: #0d2b4e; }
.cover .subtitle { font-size: 18pt; color: #2f5d92; margin: 0 0 22pt; }
.cover .lede { font-size: 12pt; line-height: 1.6; color: #33414f; max-width: 150mm; }
.cover .meta {
  margin-top: 26mm; padding-top: 10pt; border-top: 1px solid #d7e3f4;
  font-size: 9.6pt; color: #42505f;
}
.cover .meta p { margin: 0 0 5pt; }
#title-block-header { display: none; }
"""


def build_source() -> tuple[str, str]:
    chunks = []
    for name in PARTS:
        path = DOCS / name
        if not path.exists():
            raise SystemExit(f"missing guide part: {path}")
        chunks.append(path.read_text(encoding="utf-8").strip())
    source = "\n\n".join(chunks)

    lines = source.splitlines()
    cleaned = []
    dropped_title = dropped_subtitle = False
    for line in lines:
        if not dropped_title and line.strip() == "# NLP Zero to Hero":
            dropped_title = True
            continue
        if not dropped_subtitle and line.strip().startswith("## Hindi Text Restoration"):
            dropped_subtitle = True
            continue
        cleaned.append(line)
    source = "\n".join(cleaned)

    marker = "\n<div class='pagebreak'></div>\n\n# Part 0 — Orientation"
    if marker not in source:
        raise SystemExit("could not find Part 0 marker to insert the table of contents")
    source = source.replace(
        marker,
        "\n<div class='pagebreak'></div>\n\n<h1 class='toc-title'>Contents</h1>\n\n[TOC]\n\n# Part 0 — Orientation",
        1,
    )
    return source, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-out", type=Path, default=DOCS / "study_guide.html")
    parser.add_argument("--pdf-out", type=Path,
                        default=DOCS / "assblasternlp.pdf")
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()

    source, _ = build_source()
    html_body = markdown.markdown(
        source,
        extensions=["extra", "toc", "sane_lists", "codehilite"],
        extension_configs={
            "toc": {"toc_depth": "1-3", "permalink": False},
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
        },
        output_format="html5",
    )

    pygments_css = HtmlFormatter(style="default").get_style_defs(".codehilite")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NLP Zero to Hero — Study Guide</title>
<style>
{CSS}
{pygments_css}
</style>
</head>
<body>
{COVER}
{html_body}
</body>
</html>
"""
    args.html_out.parent.mkdir(parents=True, exist_ok=True)
    args.html_out.write_text(html, encoding="utf-8")
    print(f"wrote {args.html_out} ({args.html_out.stat().st_size:,} bytes)")

    if args.skip_pdf:
        return 0

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        chrome = str(mac_chrome) if mac_chrome.exists() else None
    if not chrome:
        print("Chrome/Chromium not found; HTML is ready but PDF was not built.", file=sys.stderr)
        return 1

    args.pdf_out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=30000",
        f"--print-to-pdf={args.pdf_out}",
        args.html_out.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {args.pdf_out} ({args.pdf_out.stat().st_size:,} bytes)")

    try:
        from pypdf import PdfReader

        pages = len(PdfReader(str(args.pdf_out)).pages)
        print(f"pages: {pages}")
    except Exception as exc:
        print(f"could not count pages: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
