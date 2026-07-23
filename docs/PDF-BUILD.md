# Rebuilding WHITEPAPER.pdf from WHITEPAPER.md

`WHITEPAPER.md` is the source of truth. `WHITEPAPER.pdf` (repo root) is a generated,
typeset artifact kept in the repo so the paper renders correctly everywhere — including
the GitHub mobile apps, which have never supported the `$...$`/`$$...$$` math rendering
that GitHub's desktop/web UI has shipped since 2022 (equations show as raw LaTeX source
in the app). Regenerate the PDF any time `WHITEPAPER.md` changes.

## Why this pipeline (not LaTeX/pandoc)

The build avoids a multi-gigabyte TeX toolchain by reusing two things already present in
a typical Claude Code environment: the **MathJax** npm package (for correct TeX math
typesetting — the same math syntax GitHub itself uses) and a pre-installed **headless
Chromium** (for print-to-PDF). No network access is required at render time; the MathJax
SVG output bundle is self-contained (no external font fetches).

## Recipe

```bash
# 1. Convert Markdown -> HTML, with math spans protected from Markdown's own
#    inline parsing (pymdownx.arithmatex) so underscores/braces inside $...$
#    don't get mangled by emphasis/table parsing.
pip3 install markdown pymdown-extensions

# 2. Fetch MathJax's self-contained SVG browser bundle via npm (avoids CDN,
#    which may be blocked by network policy in some environments).
mkdir -p /tmp/pdfbuild && cd /tmp/pdfbuild
npm init -y >/dev/null
npm install mathjax@3 playwright@1.56.1

# 3. Run the two build scripts (see below) to produce whitepaper.html, then
#    whitepaper.pdf via headless Chromium's page.pdf().
python3 build.py
node render.js

# 4. Copy the result to the repo root.
cp whitepaper.pdf /path/to/repo/WHITEPAPER.pdf
```

`build.py` reads `WHITEPAPER.md`, converts it with `markdown` + `pymdownx.arithmatex`
(`generic=True`) + `tables`, wraps the result in a print-styled HTML page (serif
typography, page-numbered footer via Playwright, table/equation page-break avoidance),
and inlines the MathJax bundle from `node_modules/mathjax/es5/tex-svg.js`.

`render.js` opens the HTML with Playwright's Chromium, waits for
`window.MathJax.startup.document.state() >= 10` (i.e. typesetting fully finished — do
**not** skip this wait or the PDF will capture unrendered `\(...\)` source), then calls
`page.pdf({ format: 'A4', printBackground: true, displayHeaderFooter: true, ... })`.

## A markdown gotcha this pipeline exposed (and why the source now avoids it)

Two classes of bug will silently corrupt the PDF (and can *also* silently corrupt the
GitHub-web rendering of `WHITEPAPER.md` itself, so both are worth knowing before editing):

1. **A `$$ ... $$` display-math block needs a blank line before the opening `$$` and
   after the closing `$$`.** Without it, the block merges into the surrounding paragraph
   and the math preprocessor may not recognize it — the equation leaks into the PDF as
   raw `$$ \mathrm{...} $$` text instead of being typeset. Every display block in
   `WHITEPAPER.md` is written as its own blank-line-separated 3-line unit (`$$` /
   content / `$$`).
2. **A literal `|` inside inline math (`$...$`) that sits inside a Markdown table cell
   will be parsed as a table column separator**, splitting the cell. `|V|`-style
   cardinality/absolute-value notation is therefore written as `\lvert V\rvert` in every
   table row (this is also the typographically preferred LaTeX form — plain `\|V\|`
   would be wrong too, since `\|` is the *norm* command and renders as `‖V‖`, not `|V|`).
   Outside tables, bare `$|V|$` is fine and used throughout the prose.

## Verifying a rebuild before committing

```bash
python3 -c "
import fitz
doc = fitz.open('whitepaper.pdf')
print('pages:', doc.page_count)
"
```
then visually spot-check a handful of pages (title, a proof, a table-heavy section,
references) by rasterizing with `page.get_pixmap(dpi=150)` and reading the PNGs — a
missing blank line or an unescaped table pipe is easy to introduce and easy to miss in
a text-only diff.

### Interview questions this doc answers

- *"Why is there a PDF checked in alongside a Markdown source?"* GitHub's mobile apps
  never implemented the math rendering their web UI has had since 2022; the PDF is the
  one artifact that renders identically everywhere, which matters for a document meant
  to be reviewed and published.
- *"How do you regenerate it?"* `pip install markdown pymdown-extensions`, `npm install
  mathjax playwright`, run `build.py` then `node render.js` — no LaTeX/pandoc install,
  no network access needed at render time.
- *"What's the most common way to break it?"* Missing blank lines around a `$$` block,
  or an un-escaped `|` inside inline math sitting in a table cell — both are described
  above with the fix already applied throughout the current source.
