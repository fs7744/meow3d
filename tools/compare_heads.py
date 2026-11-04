"""把多个 GLB 的**头部特写**并排成一张对比图（判断脸糊不糊只能这么看）。

    python tools/compare_heads.py out/meow3d/a.glb out/meow3d/b.glb --out out/heads.png
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BLENDER_CANDIDATES = [
    Path(r"D:\code\gen\tools\blender\blender-4.5.9-windows-x64\blender.exe"),
    Path(r"D:\GAMES_tool\blender\blender.exe"),
]


def main() -> int:
    args = sys.argv[1:]
    flat = "--flat" in args
    args = [a for a in args if a != "--flat"]
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1]).resolve()
        glbs = args[:i] + args[i + 2:]
    else:
        out = ROOT / "out" / ("heads_flat.png" if flat else "heads.png")
        glbs = args
    if not glbs:
        print(__doc__)
        return 2

    blender = next((p for p in BLENDER_CANDIDATES if p.is_file()), None)
    if blender is None:
        print("找不到 Blender")
        return 2

    heads = []
    for g in glbs:
        glb = Path(g).resolve()
        tmp = ROOT / ".tmp" / f"head_{'flat_' if flat else ''}{glb.stem}"
        tmp.mkdir(parents=True, exist_ok=True)
        p = tmp / "head.png"
        if not p.is_file():
            cmd = [str(blender), "-b", "-P",
                   str(ROOT / "tools" / "blender_preview.py"),
                   "--", str(glb), str(tmp)]
            if flat:
                cmd.append("flat")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        heads.append((glb.stem, Image.open(p).convert("RGB")))

    pad, head_h = 10, 26
    w, h = heads[0][1].size
    sheet = Image.new("RGB", (len(heads) * (w + pad) + pad, h + head_h + pad * 2),
                      (24, 25, 28))
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(heads):
        x = pad + i * (w + pad)
        d.text((x + 6, pad + 4), name, fill=(230, 232, 236))
        sheet.paste(im, (x, pad + head_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("头部对比:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
