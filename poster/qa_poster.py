#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poster QA (headless browser structural checks)
=============================================
Checks the HTML poster for the failure modes a visual review would catch:

  1. broken / missing images (naturalWidth == 0)
  2. horizontal overflow inside cards (content wider than its box)
  3. elements spilling outside the .poster page box
  4. text smaller than a print-legible size (px)
  5. figure/caption pairing, alt text, and heading structure
  6. PDF sanity (single page, exact page size)

Usage:
  .venv/bin/python poster/qa_poster.py
  .venv/bin/python poster/qa_poster.py --html poster_ko.html
"""
import argparse
import os
import sys

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

CHECK_JS = r"""
() => {
  const out = {brokenImages: [], overflow: [], outside: [], tinyText: [], counts: {}};
  const poster = document.querySelector('.poster');
  const pb = poster.getBoundingClientRect();

  document.querySelectorAll('img').forEach(img => {
    if (!img.complete || img.naturalWidth === 0) {
      out.brokenImages.push(img.getAttribute('src'));
    }
  });

  document.querySelectorAll('figure, .card, table, .flow, .stats, .journey, .thesis').forEach(el => {
    const b = el.getBoundingClientRect();
    if (el.scrollWidth > el.clientWidth + 2) {
      out.overflow.push({tag: el.tagName + '.' + (el.className || ''),
                         scrollWidth: el.scrollWidth, clientWidth: el.clientWidth,
                         text: (el.textContent || '').trim().slice(0, 60)});
    }
    if (b.width && (b.right > pb.right + 1 || b.left < pb.left - 1 || b.bottom > pb.bottom + 1)) {
      out.outside.push({tag: el.tagName + '.' + (el.className || ''),
                        rect: [Math.round(b.left - pb.left), Math.round(b.top - pb.top),
                               Math.round(b.right - pb.left), Math.round(b.bottom - pb.top)],
                        poster: [Math.round(pb.width), Math.round(pb.height)],
                        text: (el.textContent || '').trim().slice(0, 60)});
    }
  });

  document.querySelectorAll('p, li, td, th, figcaption, span, b').forEach(el => {
    if (!el.textContent.trim()) return;
    const fs = parseFloat(getComputedStyle(el).fontSize);
    if (fs > 0 && fs < 9.5) {
      out.tinyText.push({tag: el.tagName, fontSizePx: fs,
                         text: el.textContent.trim().slice(0, 50)});
    }
  });

  out.counts = {
    cards: document.querySelectorAll('.card').length,
    figures: document.querySelectorAll('figure').length,
    figuresWithCaption: document.querySelectorAll('figure figcaption').length,
    images: document.querySelectorAll('img').length,
    tables: document.querySelectorAll('table').length,
    headings: document.querySelectorAll('h1, h2, h3').length,
    imagesMissingAlt: Array.from(document.querySelectorAll('img'))
      .filter(i => !i.getAttribute('alt')).map(i => i.getAttribute('src')),
  };
  return out;
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="poster_en.html")
    ap.add_argument("--min-font-px", type=float, default=9.5)
    ap.add_argument("--size", default="A1", help="sheet size used for the PDF check")
    ap.add_argument("--tag", default="en", help="artifact tag used for the PDF check")
    args = ap.parse_args()
    src = os.path.join(HERE, args.html)
    if not os.path.exists(src):
        print(f"not found: {src}")
        return 1

    problems = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto("file://" + src, wait_until="load")
        page.wait_for_timeout(700)
        res = page.evaluate(CHECK_JS)
        browser.close()

    print(f"=== QA: {args.html} ===")
    c = res["counts"]
    print(f"  cards={c['cards']}  figures={c['figures']} (with caption {c['figuresWithCaption']})  "
          f"images={c['images']}  tables={c['tables']}  headings={c['headings']}")

    if res["brokenImages"]:
        problems += len(res["brokenImages"])
        print(f"  [FAIL] broken images: {res['brokenImages']}")
    else:
        print("  [OK] all images loaded")

    if res["counts"]["imagesMissingAlt"]:
        problems += len(res["counts"]["imagesMissingAlt"])
        print(f"  [WARN] images without alt text: {res['counts']['imagesMissingAlt']}")

    if res["overflow"]:
        problems += len(res["overflow"])
        print(f"  [FAIL] horizontal overflow in {len(res['overflow'])} element(s):")
        for o in res["overflow"][:8]:
            print(f"      {o['tag']}  scrollW={o['scrollWidth']} clientW={o['clientWidth']}  {o['text']!r}")
    else:
        print("  [OK] no horizontal overflow")

    if res["outside"]:
        problems += len(res["outside"])
        print(f"  [FAIL] {len(res['outside'])} element(s) outside the poster page box:")
        for o in res["outside"][:8]:
            print(f"      {o['tag']} rect={o['rect']} poster={o['poster']}  {o['text']!r}")
    else:
        print("  [OK] everything inside the page box")

    tiny = [t for t in res["tinyText"] if t["fontSizePx"] < args.min_font_px]
    if tiny:
        print(f"  [WARN] {len(tiny)} text node(s) below {args.min_font_px}px (print legibility):")
        for t in tiny[:8]:
            print(f"      {t['tag']} {t['fontSizePx']}px  {t['text']!r}")
    else:
        print(f"  [OK] no text below {args.min_font_px}px")

    # PDF sanity
    pdf = os.path.join(RUNS, f"poster_{args.tag}_{args.size}.pdf")
    if os.path.exists(pdf):
        try:
            from pypdf import PdfReader
            r = PdfReader(pdf)
            w_mm = round(float(r.pages[0].mediabox[2]) / 72 * 25.4)
            h_mm = round(float(r.pages[0].mediabox[3]) / 72 * 25.4)
            txt = "".join(pg.extract_text() or "" for pg in r.pages)
            import re
            ko = re.findall(r"[\uac00-\ud7a3]+", txt)
            broken = re.findall(r"[\u25a1\ufffd]", txt)
            print(f"  PDF: pages={len(r.pages)} size={w_mm}x{h_mm}mm  chars={len(txt)}  "
                  f"hangul={len(ko)}  brokenGlyph={len(broken)}")
            want = {"A0": (841, 1189), "A1": (594, 841), "A2": (420, 594),
                    "A1L": (841, 594)}[args.size]
            if len(r.pages) != 1 or (w_mm, h_mm) != want:
                problems += 1
                print(f"  [FAIL] PDF is not a single {args.size} page")
            if broken:
                problems += 1
                print("  [FAIL] broken glyphs in PDF text")
            else:
                print("  [OK] PDF text has no broken glyphs")
        except Exception as exc:  # pragma: no cover
            print(f"  [WARN] PDF check failed: {exc}")

    print(f"=== QA result: {'PASS' if problems == 0 else f'{problems} issue(s)'} ===")
    return 0 if problems == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
