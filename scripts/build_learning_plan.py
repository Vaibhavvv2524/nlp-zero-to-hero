#!/usr/bin/env python3
"""Build the priority-order learning guide (docs/what-to-learn-priority-order.pdf)."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SOURCE_DIR = DOCS / "learning"

COVER = """
<section class="cover">
  <div class="kicker">Priority map &middot; study less, know more</div>
  <h1>WHAT TO<br>LEARN</h1>
  <p class="subtitle">The short, ordered path through the Hindi punctuation &rarr;
  sentence ordering &rarr; translation project.</p>
  <p class="lede">Start with the story. Add one stage at a time. Stop when you can
  explain the pipeline, defend its weak scores, and name what you must not claim.</p>
  <div class="meta">
    <p><strong>Order:</strong> P0 story &rarr; P1 punctuation &rarr; P2 ordering &rarr; P3 translation &rarr; P4-P7 depth</p>
    <p><strong>Minimum:</strong> 30 minutes for the story and Stage 1; 2 hours for the scoreboard</p>
    <p><strong>Evidence:</strong> subset adapters &middot; boundary F1 0.884 &middot; Borda tau 0.201 &middot; BLEU 34.1</p>
    <p><strong>Repository:</strong> github.com/Vaibhavvv2524/nlp-zero-to-hero</p>
  </div>
  <div class="stamp">P0 &middot; P1 &middot; P2 &middot; P3 FIRST</div>
</section>
"""

CSS = """
@page { size: A4; margin: 14mm 14mm 14mm 14mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Charter", "Georgia", "Devanagari Sangam MN", "Noto Serif Devanagari", serif;
  font-size: 10.0pt; line-height: 1.48; color: #17202b; margin: 0;
}
h1, h2, h3, h4 {
  font-family: "Avenir Next", "Helvetica Neue", "Devanagari Sangam MN", sans-serif;
  color: #0d2b4e; page-break-after: avoid; break-after: avoid;
}
h1 {
  font-size: 20pt; margin: 20pt 0 8pt; padding-bottom: 5pt;
  border-bottom: 2.5px solid #c9a227; letter-spacing: -0.2pt;
}
.cover h1, .toc-title { page-break-before: avoid; break-before: auto; }
h2 { font-size: 13.2pt; margin: 15pt 0 5pt; color: #12355b; }
h3 { font-size: 11.2pt; margin: 12pt 0 4pt; color: #1d4f7a; }
h4 { font-size: 10.3pt; margin: 11pt 0 3pt; }
p { margin: 0 0 5.5pt; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 8pt; padding-left: 16pt; }
li { margin: 0 0 3pt; }
a { color: #12355b; text-decoration: none; }
strong { color: #0b2540; }
em { color: #243447; }
hr { border: 0; border-top: 1px solid #e3e8ef; margin: 14pt 0; }
blockquote {
  margin: 10pt 0; padding: 8pt 11pt; background: #f5f7fa;
  border-left: 3px solid #c9a227; color: #2b3644; page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
code {
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 8.5pt; background: #f2f4f7; padding: 0.5pt 3pt; border-radius: 2pt;
  color: #1f3550;
}
pre { margin: 8pt 0 10pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.0pt; line-height: 1.42; }
.codehilite {
  background: #f6f8fa; border: 1px solid #e3e8ef; border-radius: 3pt;
  padding: 7pt 9pt; overflow: hidden; page-break-inside: avoid;
}
table {
  width: 100%; border-collapse: collapse; margin: 8pt 0 12pt;
  font-size: 8.4pt; page-break-inside: auto;
}
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #d8dee7; padding: 3.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #eaf0f8; color: #12355b; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafbfd; }
.lead {
  margin: 12pt 0 16pt; padding: 10pt 13pt; background: #f3f7fd;
  border: 1px solid #dbe6f5; border-radius: 4pt; font-size: 11.2pt;
  line-height: 1.55; color: #14294a;
}
.lead p:last-child { margin-bottom: 0; }
.callout, .pass, .priority, .warning {
  margin: 10pt 0 12pt; padding: 8pt 11pt; border-radius: 3pt;
  page-break-inside: avoid; font-size: 9.8pt; line-height: 1.5;
}
.callout { background: #eef4ff; border-left: 4px solid #2563eb; color: #14294a; }
.pass { background: #f2fbf5; border-left: 4px solid #1f8a4c; color: #143522; }
.priority { background: #fff7ea; border-left: 4px solid #c47b12; color: #3d2a08; }
.warning { background: #fdecec; border-left: 4px solid #b91c1c; color: #3b1212; }
.callout p:last-child, .pass p:last-child, .priority p:last-child,
.warning p:last-child { margin-bottom: 0; }
.formula {
  margin: 8pt 0 11pt; padding: 7pt 10pt; background: #f7f9fc;
  border: 1px solid #e1e7f0; border-radius: 3pt;
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
  font-size: 10pt; color: #12355b; text-align: center;
  page-break-inside: avoid;
}
.pagebreak { display: block; height: 0; margin: 0; padding: 0;
  page-break-after: always; break-after: page; }
figure.diagram {
  margin: 9pt 0 14pt; padding: 6pt 6pt 5pt; background: #fbfcfe;
  border: 1px solid #dbe3ee; border-radius: 4pt; page-break-inside: avoid;
  break-inside: avoid;
}
figure.diagram img { display: block; width: 96%; height: auto; margin: 0 auto; }
figure.diagram figcaption {
  margin-top: 5pt; font-size: 8.4pt; line-height: 1.4; color: #475569;
  font-family: "Avenir Next", "Helvetica Neue", sans-serif;
}
figure.diagram figcaption strong { color: #0d2b4e; }
.toc-title { margin-bottom: 8pt; }
.toc { font-size: 9.6pt; column-count: 2; column-gap: 9mm;
  page-break-after: always; break-after: page; }
.toc > ul { list-style: none; padding-left: 0; margin: 0; }
.toc ul { list-style: none; padding-left: 9pt; margin: 2pt 0; }
.toc li { margin: 0 0 3pt; break-inside: avoid; }
.toc > ul > li { font-weight: 600; color: #12355b; margin-top: 6pt; }
.toc a { color: inherit; }
.cover {
  margin: 0; padding: 24mm 18mm 16mm; min-height: 257mm;
  background: #0b2340; color: #f8fafc; page-break-after: always; break-after: page;
}
.cover .kicker {
  font-family: "Avenir Next", "Helvetica Neue", sans-serif; font-size: 8.6pt;
  letter-spacing: 2.4pt; text-transform: uppercase; color: #d6b85a; margin-bottom: 14pt;
}
.cover h1 {
  font-family: "Avenir Next", "Helvetica Neue", sans-serif;
  font-size: 58pt; line-height: 0.94; border: none; margin: 0 0 16pt;
  color: #ffffff; letter-spacing: -1.4pt;
}
.cover .subtitle { font-size: 17pt; line-height: 1.35; color: #cfe0f5; margin: 0 0 18pt; max-width: 155mm; }
.cover .lede { font-size: 11.4pt; line-height: 1.6; color: #e7eef8; max-width: 150mm; }
.cover .meta {
  margin-top: 38mm; padding-top: 10pt; border-top: 1px solid #3b5577;
  font-size: 9.2pt; line-height: 1.5; color: #dbe7f5;
}
.cover .meta p { margin: 0 0 5pt; }
.cover .stamp {
  margin-top: 12mm; display: inline-block; padding: 4pt 9pt;
  border: 1px solid #d6b85a; color: #d6b85a; font-size: 8pt;
  letter-spacing: 1.8pt; font-family: "Avenir Next", "Helvetica Neue", sans-serif;
}
#title-block-header { display: none; }
"""


def build_source() -> str:
    paths = sorted(SOURCE_DIR.glob("*.md"))
    if not paths:
        raise SystemExit(f"no learning source files in {SOURCE_DIR}")
    chunks = []
    for path in paths:
        chunks.append(path.read_text(encoding="utf-8").strip())
    source = "\n\n".join(chunks)
    source = re.sub(
        r'<div class="(lead|callout|pass|priority|warning)">',
        r'<div class="\1" markdown="1">',
        source,
    )
    source = '<h1 class="toc-title">Contents</h1>\n\n[TOC]\n\n' + source
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-out", type=Path, default=DOCS / "what-to-learn-priority-order.html")
    parser.add_argument("--pdf-out", type=Path, default=DOCS / "what-to-learn-priority-order.pdf")
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()

    source = build_source()
    html_body = markdown.markdown(
        source,
        extensions=["extra", "md_in_html", "toc", "sane_lists", "codehilite"],
        extension_configs={
            "toc": {"toc_depth": "1-1", "permalink": False},
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
<title>What to Learn, in Priority Order - Hindi Restoration and Translation</title>
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

        print(f"pages: {len(PdfReader(str(args.pdf_out)).pages)}")
    except Exception as exc:
        print(f"could not count pages: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
