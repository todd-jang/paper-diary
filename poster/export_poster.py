#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTML poster -> print-ready PDF / PNG
====================================
Renders `poster_en.html` (or any local HTML) with headless Chromium and exports:
  * PDF at the exact poster page size (A0 by default: 841 x 1189 mm)
  * a high-resolution PNG preview

Usage:
  .venv/bin/python poster/export_poster.py                 # A0 PDF + PNG
  .venv/bin/python poster/export_poster.py --size A1       # A1 (594 x 841 mm)
  .venv/bin/python poster/export_poster.py --html poster_ko.html --tag ko
  .venv/bin/python poster/export_poster.py --png-only      # PNG preview only

Requires: playwright + chromium (`.venv/bin/python -m playwright install chromium`).
"""
import argparse
import os

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)

PAGE_MM = {
    "A0": (841.0, 1189.0),
    "A1": (594.0, 841.0),
    "A2": (420.0, 594.0),
    "A1L": (841.0, 594.0),   # landscape sheet for the space-themed poster
}
MM_TO_PX = 96.0 / 25.4


def export(html_name: str, size: str, tag: str, png_only: bool) -> list:
    w_mm, h_mm = PAGE_MM[size]
    src = os.path.join(HERE, html_name)
    if not os.path.exists(src):
        raise SystemExit(f"not found: {src}")

    out_pdf = os.path.join(RUNS, f"poster_{tag}_{size}.pdf")
    out_png = os.path.join(RUNS, f"poster_{tag}_{size}.png")
    poster_scale = w_mm / PAGE_MM["A0"][0]
    a1_layout = """
        .poster { min-height: 0 !important; height: auto !important; gap: 8.5mm !important; }
        .hero { padding-top: 18mm !important; padding-bottom: 16mm !important; }
        .journey { padding-top: 5.5mm !important; padding-bottom: 5.5mm !important; }
        .card { padding: 8mm !important; }
        h2 { font-size: 25pt !important; margin-bottom: 5mm !important; }
        h3 { font-size: 15.5pt !important; }
        p, li { font-size: 13pt !important; line-height: 1.62 !important; }
        table { font-size: 11pt !important; }
        th { font-size: 9.5pt !important; }
        figure { margin-top: 5mm !important; }
        figure img { max-height: 140mm !important; }
        .column, .wide { gap: 7mm !important; }
        .columns { flex: none !important; }
    """ if size == "A1" else ""

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": int(w_mm * MM_TO_PX), "height": int(h_mm * MM_TO_PX)},
                                device_scale_factor=1)
        page.goto("file://" + src, wait_until="load")
        if size == "A1L":
            # landscape design carries its own page box (841x594mm) - do not force the A0 box
            page.add_style_tag(content="""
                html, body { margin: 0 !important; padding: 0 !important; }
                .poster { margin: 0 !important; box-shadow: none !important; }
            """)
        else:
            page.add_style_tag(content=f"""
                html, body {{ margin: 0 !important; padding: 0 !important; }}
                .poster {{ margin: 0 !important; box-shadow: none !important;
                           width: {PAGE_MM["A0"][0]}mm !important;
                           min-height: {PAGE_MM["A0"][1]}mm !important;
                           zoom: {poster_scale:.8f} !important; }}
            """)
        if a1_layout:
            page.add_style_tag(content=a1_layout)
        page.wait_for_timeout(600)  # let fonts/images settle
        if not png_only:
            page.pdf(path=out_pdf, width=f"{w_mm}mm", height=f"{h_mm}mm",
                     print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
                     prefer_css_page_size=False)
            print("  saved", os.path.relpath(out_pdf, os.path.dirname(HERE)))
        page.screenshot(path=out_png, full_page=True, scale="css")
        print("  saved", os.path.relpath(out_png, os.path.dirname(HERE)))
        browser.close()
    return [out_pdf, out_png]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="poster_en.html")
    ap.add_argument("--size", default="A1", choices=list(PAGE_MM))
    ap.add_argument("--tag", default="en")
    ap.add_argument("--png-only", action="store_true")
    args = ap.parse_args()
    print(f"[export] {args.html} -> {args.size} ({PAGE_MM[args.size][0]}x{PAGE_MM[args.size][1]} mm)")
    export(args.html, args.size, args.tag, args.png_only)


if __name__ == "__main__":
    main()
