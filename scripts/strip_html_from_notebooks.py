"""Strip raw HTML from Jupyter notebook markdown cells.

Processes three hand-maintained notebooks in-place:
  - notebooks/cosPaper_Replication.ipynb
  - notebooks/fo2008_replication.ipynb
  - notebooks/demo.ipynb

Run from repo root:
    python3 scripts/strip_html_from_notebooks.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    ROOT / "notebooks" / "cosPaper_Replication.ipynb",
    ROOT / "notebooks" / "fo2008_replication.ipynb",
    ROOT / "notebooks" / "demo.ipynb",
]


# ---------------------------------------------------------------------------
# Utility: extract the full extent of a balanced <div>…</div> block
# ---------------------------------------------------------------------------

def _find_balanced_div(s: str, start: int) -> tuple[int, int] | None:
    """Return (start, end) indices of the balanced <div>…</div> starting at `start`.

    `start` should point to the '<' of the opening <div...>.
    Returns None if no balanced close is found.
    """
    depth = 0
    i = start
    n = len(s)
    while i < n:
        open_m = re.search(r'<div\b', s[i:], re.IGNORECASE)
        close_m = re.search(r'</div>', s[i:], re.IGNORECASE)
        if open_m is None and close_m is None:
            return None
        if open_m is None:
            ci = i + close_m.start()
        elif close_m is None:
            ci = None
            break
        else:
            oi = i + open_m.start()
            ci_c = i + close_m.start()
            if oi < ci_c:
                depth += 1
                i = oi + 1
                continue
            else:
                ci = ci_c
        # ci is a closing div
        depth -= 1
        if depth == 0:
            end = ci + len("</div>")
            return (start, end)
        i = ci + 1
    return None


def _extract_nested_div_body(s: str, start: int) -> tuple[str, int] | None:
    """Return (inner_body, end_index) for the div starting at `start`."""
    result = _find_balanced_div(s, start)
    if result is None:
        return None
    div_start, div_end = result
    # Find position right after the opening tag's '>'
    gt = s.index('>', div_start)
    inner = s[gt + 1: div_end - len('</div>')]
    return inner, div_end


# ---------------------------------------------------------------------------
# Low-level HTML helpers
# ---------------------------------------------------------------------------

def _strip_tags(s: str) -> str:
    """Strip all HTML tags, keeping inner text."""
    return re.sub(r'<[^>]+>', '', s)


# ---------------------------------------------------------------------------
# Rule 1: Hero banner  <div style="background:linear-gradient...">…</div>
# ---------------------------------------------------------------------------

def _convert_hero(s: str) -> str:
    """Convert the top-level hero <div style="background:linear-gradient…"> block."""
    hero_start = re.search(r'<div\s+style="background:linear-gradient', s)
    if not hero_start:
        return s

    start = hero_start.start()
    result = _extract_nested_div_body(s, start)
    if result is None:
        return s
    body, end = result

    # Extract supertitle (small "Columbia University…" line, font-size 12px or 11px)
    supertitle_m = re.search(
        r'<div\s[^>]*font-size:\s*1[123]px[^>]*>(.*?)</div>', body, re.DOTALL
    )
    supertitle = ""
    if supertitle_m:
        supertitle = re.sub(r'\s+', ' ', _strip_tags(supertitle_m.group(1))).strip()

    # Extract h1
    h1_m = re.search(r'<h1[^>]*>(.*?)</h1>', body, re.DOTALL)
    title = re.sub(r'\s+', ' ', _strip_tags(h1_m.group(1))).strip() if h1_m else ""

    # Extract subtitle div (font-size 15px or 16px)
    subtitle_m = re.search(
        r'<div\s[^>]*font-size:\s*1[456]px[^>]*>(.*?)</div>', body, re.DOTALL
    )
    subtitle = ""
    if subtitle_m:
        raw = subtitle_m.group(1)
        raw = re.sub(r'<br\s*/?>', ' ', raw)
        subtitle = re.sub(r'\s+', ' ', _strip_tags(raw)).strip()

    # Extract meta line (font-size 13px)
    meta_m = re.search(
        r'<div\s[^>]*font-size:\s*13px[^>]*>(.*?)</div>', body, re.DOTALL
    )
    meta = ""
    if meta_m:
        raw = meta_m.group(1)
        raw = re.sub(r'<br\s*/?>', ' ', raw)
        # convert inline code
        raw = re.sub(r'<code[^>]*>(.*?)</code>', lambda m: f'`{_strip_tags(m.group(1))}`', raw, flags=re.DOTALL)
        meta = re.sub(r'\s+', ' ', _strip_tags(raw)).strip()

    lines = []
    if title:
        lines.append(f"# {title}")
    if supertitle:
        lines.append(f"\n**{supertitle}**")
    if subtitle:
        lines.append(f"\n*{subtitle}*")
    if meta:
        lines.append(f"\n*{meta}*")
    lines.append("\n---")

    replacement = "\n".join(lines)
    return s[:start] + replacement + s[end:]


# ---------------------------------------------------------------------------
# Rule 2: <style> blocks → delete
# ---------------------------------------------------------------------------

def _remove_style_blocks(s: str) -> str:
    return re.sub(r'<style[^>]*>.*?</style>', '', s, flags=re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Rule 3: HTML comments → delete
# ---------------------------------------------------------------------------

def _remove_html_comments(s: str) -> str:
    return re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Rule 4/5: <div class="cu-card…"> blocks
# ---------------------------------------------------------------------------

def _convert_cu_cards(s: str) -> str:
    """Convert cu-card divs to markdown blockquotes (handles nesting iteratively)."""
    for _ in range(20):
        # Find the LAST (innermost) opening cu-card div
        m = None
        for candidate in re.finditer(r'<div\s+class="cu-card[^"]*"[^>]*>', s):
            m = candidate
        if m is None:
            break

        start = m.start()
        result = _extract_nested_div_body(s, start)
        if result is None:
            break
        inner, end = result

        class_val = m.group(0)
        warn = 'cu-warn' in class_val or 'cu-anomaly' in class_val
        replacement = _card_to_blockquote(inner, warn=warn)
        s = s[:start] + replacement + s[end:]

    return s


def _card_to_blockquote(inner: str, *, warn: bool = False) -> str:
    """Convert the inner content of a cu-card div to a markdown blockquote."""
    lines = []

    # Extract optional <span class="cu-flag">
    flag_m = re.search(r'<span\s+class="cu-flag"[^>]*>(.*?)</span>', inner, re.DOTALL)
    flag_text = ""
    if flag_m:
        flag_text = re.sub(r'\s+', ' ', _strip_tags(flag_m.group(1))).strip()
        inner = inner[:flag_m.start()] + inner[flag_m.end():]

    # Extract <h3>
    h3_m = re.search(r'<h3[^>]*>(.*?)</h3>', inner, re.DOTALL)
    heading = ""
    if h3_m:
        heading = re.sub(r'\s+', ' ', _strip_tags(h3_m.group(1))).strip()
        inner = inner[:h3_m.start()] + inner[h3_m.end():]

    # Compose heading line
    if heading or flag_text:
        prefix = "⚠️ " if warn else ""
        if flag_text and heading:
            lines.append(f"> **{prefix}{flag_text} — {heading}**")
        elif flag_text:
            lines.append(f"> **{prefix}{flag_text}**")
        else:
            lines.append(f"> **{prefix}{heading}**")
        lines.append(">")

    # Extract <p> tags
    for p_m in re.finditer(r'<p[^>]*>(.*?)</p>', inner, re.DOTALL):
        para = p_m.group(1)
        # Handle <br/> within paragraphs
        para = re.sub(r'<br\s*/?>', '  \n> ', para)
        # Handle inline tags within paragraph
        para = _convert_inline_tags(para)
        para_text = re.sub(r'\s+', ' ', _strip_tags(para)).strip()
        if para_text:
            lines.append(f"> {para_text}")
            lines.append(">")

    # Extract <ul><li> lists
    for ul_m in re.finditer(r'<ul[^>]*>(.*?)</ul>', inner, re.DOTALL):
        for li_m in re.finditer(r'<li[^>]*>(.*?)</li>', ul_m.group(1), re.DOTALL):
            li_text = re.sub(r'\s+', ' ', _strip_tags(_convert_inline_tags(li_m.group(1)))).strip()
            lines.append(f"> - {li_text}")
        lines.append(">")

    # Remove trailing "> " blank lines
    while lines and lines[-1] == ">":
        lines.pop()

    return "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Rule 6: Footer <div style="…background:#012169…">text</div>
# ---------------------------------------------------------------------------

def _convert_footer_div(s: str) -> str:
    """Convert footer divs with dark Columbia blue background."""
    # Find div style= blocks that look like footer (background: dark blue)
    for m in list(re.finditer(r'<div\s+style="[^"]*"[^>]*>', s)):
        # Check if this is a footer-style div (background with dark hex colour)
        tag = m.group(0)
        if not re.search(r'background:\s*#0[0-9A-Fa-f]{5}', tag):
            continue
        start = m.start()
        result = _extract_nested_div_body(s, start)
        if result is None:
            continue
        inner, end = result
        text = re.sub(r'\s+', ' ', _strip_tags(inner)).strip()
        s = s[:start] + f"\n---\n*{text}*\n" + s[end:]
        # Only process one at a time to avoid index issues; re-run via caller
        break
    return s


def _convert_footer_divs(s: str) -> str:
    """Iteratively convert all footer divs."""
    for _ in range(10):
        new_s = _convert_footer_div(s)
        if new_s == s:
            break
        s = new_s
    return s


# ---------------------------------------------------------------------------
# Inline tag conversions (rules 7–12)
# ---------------------------------------------------------------------------

def _convert_inline_tags(s: str) -> str:
    """Convert inline HTML tags to markdown equivalents."""
    # Rule 7: cu-flag spans
    s = re.sub(
        r'<span\s+class="cu-flag"[^>]*>(.*?)</span>',
        lambda m: f"`{re.sub(r'\\s+', ' ', _strip_tags(m.group(1))).strip()}`",
        s, flags=re.DOTALL
    )
    # Rule 8: inline <code>
    s = re.sub(
        r'<code[^>]*>(.*?)</code>',
        lambda m: f"`{_strip_tags(m.group(1))}`",
        s, flags=re.DOTALL
    )
    # Rule 9: <b> / <strong>
    s = re.sub(
        r'<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>',
        lambda m: f"**{re.sub(r'\\s+', ' ', _strip_tags(m.group(1))).strip()}**",
        s, flags=re.DOTALL
    )
    # Rule 10: <em> / <i>
    s = re.sub(
        r'<(?:em|i)[^>]*>(.*?)</(?:em|i)>',
        lambda m: f"*{re.sub(r'\\s+', ' ', _strip_tags(m.group(1))).strip()}*",
        s, flags=re.DOTALL
    )
    # Rule 11: <br/> or <br>
    s = re.sub(r'<br\s*/?>', '  \n', s)
    # Rule 12: <sup>
    s = re.sub(
        r'<sup[^>]*>(.*?)</sup>',
        lambda m: f"^{_strip_tags(m.group(1))}^",
        s, flags=re.DOTALL
    )
    return s


# ---------------------------------------------------------------------------
# Rule 13: strip any remaining tags + decode entities
# ---------------------------------------------------------------------------

def _strip_remaining_tags(s: str) -> str:
    """Strip any remaining HTML tags, decode HTML entities."""
    s = s.replace('&amp;', '&')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&quot;', '"')
    s = s.replace('&nbsp;', ' ')
    # Strip remaining tags
    s = re.sub(r'<[^>]+>', '', s)
    return s


# ---------------------------------------------------------------------------
# Rule 14: Normalize whitespace
# ---------------------------------------------------------------------------

def _normalize_whitespace(s: str) -> str:
    """Collapse 3+ blank lines to 2, strip leading/trailing whitespace."""
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


# ---------------------------------------------------------------------------
# Special: <div class="cu-figure-card"> with <img>
# ---------------------------------------------------------------------------

def _convert_figure_card(s: str) -> str:
    """Convert cu-figure-card divs containing img tags."""
    for m in list(re.finditer(r'<div\s+class="cu-figure-card"[^>]*>', s)):
        start = m.start()
        result = _extract_nested_div_body(s, start)
        if result is None:
            continue
        inner, end = result
        img_m = re.search(
            r'<img\s[^>]*src="([^"]*)"(?:[^>]*alt="([^"]*)")?[^>]*/?>',
            inner, re.DOTALL
        )
        if img_m:
            src = img_m.group(1)
            alt = img_m.group(2) or ""
            replacement = f"![{alt}]({src})"
        else:
            replacement = ""
        s = s[:start] + replacement + s[end:]
        break  # re-run iteratively
    return s


def _convert_figure_cards(s: str) -> str:
    for _ in range(10):
        new_s = _convert_figure_card(s)
        if new_s == s:
            break
        s = new_s
    return s


# ---------------------------------------------------------------------------
# Master transform function
# ---------------------------------------------------------------------------

def transform_markdown(s: str) -> str:
    """Apply all HTML → markdown transformations to a markdown cell source."""
    # Rule 3: Remove HTML comments first
    s = _remove_html_comments(s)
    # Rule 2: Remove style blocks
    s = _remove_style_blocks(s)
    # Rule 1: Hero banner (must come before cu-card processing)
    s = _convert_hero(s)
    # Special: figure cards
    s = _convert_figure_cards(s)
    # Rule 4/5: cu-card blocks
    s = _convert_cu_cards(s)
    # Rule 6: Footer divs
    s = _convert_footer_divs(s)
    # Apply inline conversions
    s = _convert_inline_tags(s)
    # Rule 13: Strip remaining tags + decode entities
    s = _strip_remaining_tags(s)
    # Rule 14: Normalize whitespace
    s = _normalize_whitespace(s)
    return s


# ---------------------------------------------------------------------------
# Notebook processing
# ---------------------------------------------------------------------------

def process_notebook(nb_path: Path) -> int:
    """Process a notebook in-place. Returns count of cells modified."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    modified = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        if isinstance(cell["source"], list):
            src = "".join(cell["source"])
        else:
            src = cell["source"]

        new_src = transform_markdown(src)

        if new_src != src:
            if new_src:
                lines = new_src.splitlines(keepends=True)
                # Ensure last line has no trailing newline
                if lines and lines[-1].endswith("\n"):
                    lines[-1] = lines[-1].rstrip("\n")
                cell["source"] = lines
            else:
                cell["source"] = []
            modified += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return modified


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    for nb_path in NOTEBOOKS:
        if not nb_path.exists():
            print(f"SKIP (not found): {nb_path}")
            continue
        n = process_notebook(nb_path)
        print(f"processed {nb_path.name}: {n} cell(s) modified")


if __name__ == "__main__":
    main()
