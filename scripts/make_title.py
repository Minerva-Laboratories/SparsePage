"""Title artwork for the README (a wordmark, no logo), light and dark: docs/assets/<stem>-{light,dark}.svg / .png.

The motif under the wordmark is the method itself, drawn from its real schedule:
  sparsepage  a row of KV pages, the opened top ~15 % lit (score -> select -> read only those);
  depthskip   a depth profile per prompt token for the graded g512 plan: old tokens stop at layer 16, the last 512 at
              24, the last 32 run every layer (bar height = layers computed).
Text is converted to outline paths (IBM Plex Sans through fontTools), so the SVGs need no installed font; PNGs are
rendered with a headless Chromium (TITLE_CHROME, default Playwright's chrome-headless-shell, else google-chrome).

    python scripts/make_title.py sparsepage|depthskip [--font IBMPlexSans[wdth,wght].ttf]
"""
import argparse
import glob
import os
import pathlib
import random
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
INK = {"light": "#1f2328", "dark": "#e6edf3"}
MUTED = {"light": "#d0d7de", "dark": "#30363d"}
SPEC = {
    "sparsepage": dict(words=("Sparse", "Page"), accent={"light": ["#0f6f9e", "#1d9bd1", "#5cc8f0"],
                                                         "dark": ["#2a8fc4", "#4db6e8", "#8fd9f7"]}),
    "depthskip": dict(words=("Depth", "Skip"), accent={"light": ["#b4501d", "#e07b39", "#f3a861"],
                                                      "dark": ["#d0662a", "#f08c4a", "#ffbe82"]}),
}


def outline(font_path, text, size, x, y, tracking=-2.0, weight=600):
    """text -> SVG path data (outlines) with the baseline at y; returns (d, advance width)."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont
    font = TTFont(font_path)
    if "fvar" in font:
        from fontTools.varLib.instancer import instantiateVariableFont
        font = instantiateVariableFont(font, {a.axisTag: (weight if a.axisTag == "wght" else a.defaultValue)
                                              for a in font["fvar"].axes})
    gs, cmap = font.getGlyphSet(), font.getBestCmap()
    s = size / font["head"].unitsPerEm
    pen, cx = SVGPathPen(gs), x
    for ch in text:
        g = cmap[ord(ch)]
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, cx, y)))
        cx += gs[g].width * s + tracking
    return pen.getCommands(), cx - tracking - x


def motif(kind, mode, x0, width, y0):
    acc = SPEC[kind]["accent"][mode]
    out = []
    if kind == "sparsepage":
        n, gap = 40, 4
        w = (width - gap * (n - 1)) / n
        rng = random.Random(7)
        opened = {0} | set(rng.sample(range(1, n), 6))          # page 0 (sinks) always opens, ~15 % in all
        for i in range(n):
            x = x0 + i * (w + gap)
            fill = "url(#acc)" if i in opened else MUTED[mode]
            out.append(f'<rect x="{x:.1f}" y="{y0}" width="{w:.1f}" height="22" rx="3" fill="{fill}"/>')
    else:
        n, gap = 64, 2
        w = (width - gap * (n - 1)) / n
        full = 30
        for i in range(n):                                       # token position left (old) -> right (newest)
            frac = 16 / 32 if i < n - 12 else (24 / 32 if i < n - 3 else 1.0)
            h = full * frac
            x = x0 + i * (w + gap)
            fill = MUTED[mode] if frac < 0.6 else ("url(#acc)")
            op = "" if frac < 1 else ""
            out.append(f'<rect x="{x:.1f}" y="{y0 + full - h:.1f}" width="{w:.1f}" height="{h:.1f}" rx="1.5" '
                       f'fill="{fill}"{op}/>')
    return "".join(out)


def svg(kind, mode, font, scale=1):
    a, b = SPEC[kind]["words"]
    acc = SPEC[kind]["accent"][mode]
    d1, w1 = outline(font, a, 104, 24, 118)
    d2, w2 = outline(font, b, 104, 24 + w1 + 2, 118)
    W = int(24 + w1 + 2 + w2 + 24)
    H = 196
    stops = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in zip((0, .5, 1), acc))
    body = (f'<defs><linearGradient id="acc" x1="0" y1="0" x2="1" y2="1">{stops}</linearGradient></defs>'
            f'<path d="{d1}" fill="{INK[mode]}"/><path d="{d2}" fill="url(#acc)"/>'
            + motif(kind, mode, 26, W - 52, 148))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * scale}" '
            f'height="{H * scale}">{body}</svg>\n'), W, H


def chrome():
    env = os.environ.get("TITLE_CHROME")
    if env:
        return env
    found = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"))
                   + glob.glob("/mnt/ext/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"))
    return found[-1] if found else "google-chrome"


def render(text, png, w, h):
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as f:
        f.write(text)
        src = f.name
    subprocess.run([chrome(), "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    "--default-background-color=00000000", f"--window-size={w},{h}", f"--screenshot={png}",
                    f"file://{src}"], check=True, capture_output=True, timeout=120)
    os.unlink(src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(SPEC))
    ap.add_argument("--font", default=os.path.expanduser("~/.local/share/fonts/IBMPlexSans-Variable.ttf"))
    a = ap.parse_args()
    out = ROOT / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        text, W, H = svg(a.kind, mode, a.font)
        (out / f"{a.kind}-{mode}.svg").write_text(text)
        render(svg(a.kind, mode, a.font, 3)[0], str((out / f"{a.kind}-{mode}.png").resolve()), W * 3, H * 3)
        print("wrote", out / f"{a.kind}-{mode}.png")
