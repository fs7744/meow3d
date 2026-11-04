"""渲染 GLB 四视角预览并拼成一张图。

    python tools/preview_glb.py out/meow3d/xxx.glb
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
NAMES = ["front", "three_quarter", "side", "back"]
LABELS = ["正面", "3/4 侧", "正侧", "背面"]


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python tools/preview_glb.py <模型.glb>")
        return 2
    glb = Path(sys.argv[1]).resolve()
    if not glb.is_file():
        print("找不到", glb)
        return 2

    blender = next((p for p in BLENDER_CANDIDATES if p.is_file()), None)
    if blender is None:
        print("找不到 Blender，改 BLENDER_CANDIDATES")
        return 2

    tmp = ROOT / ".tmp" / f"preview_{glb.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(blender), "-b", "-P", str(ROOT / "tools" / "blender_preview.py"),
         "--", str(glb), str(tmp)],
        check=True,
    )

    imgs = []
    for n in NAMES:
        p = tmp / f"{n}.png"
        if not p.is_file():
            print("缺视角", p)
            continue
        imgs.append(Image.open(p).convert("RGB"))
    if not imgs:
        return 1

    w, h = imgs[0].size
    cols = 2
    rows = (len(imgs) + cols - 1) // cols
    pad, head = 12, 28
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + head + pad) + pad),
                      (24, 25, 28))
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        x = pad + (i % cols) * (w + pad)
        y = pad + (i // cols) * (h + head + pad)
        d.text((x + 6, y + 6), LABELS[i], fill=(230, 232, 236))
        sheet.paste(im, (x, y + head))
    out = ROOT / "out" / f"{glb.stem}_views.png"
    sheet.save(out)
    print("预览:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
