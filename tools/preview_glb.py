"""渲染 GLB 四视角预览并拼成一张图。

    python tools/preview_glb.py out/meow3d/xxx.glb
    python tools/preview_glb.py out/meow3d/xxx.glb --flat

`--flat` 把 Base Color 直连 Emission、不打灯，用来判断**贴图本色**。
带灯光的预览对低 roughness 的 PBR 材质会打出过曝白斑（2026-11-05 Asuka
实测：roughness 39/255 时正/背面整片死白，FLAT 下完全正常），所以判断颜色
必须看 FLAT 版；灯光版只用来看立体感与轮廓。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
BLENDER_CANDIDATES = [
    Path(r"D:\GAMES_tool\blender-5.4.0-alpha+main.6f4fc760c138-windows.amd64-release\blender.exe"),
    Path(r"D:\code\gen\tools\blender\blender-4.5.9-windows-x64\blender.exe"),
    Path(r"D:\GAMES_tool\blender\blender.exe"),
]
NAMES = ["front", "three_quarter", "side", "back"]
LABELS = ["正面", "3/4 侧", "正侧", "背面"]


def main() -> int:
    argv = sys.argv[1:]
    flat = "--flat" in argv
    pos = [a for a in argv if not a.startswith("--")]
    if not pos:
        print("用法: python tools/preview_glb.py <模型.glb> [--flat]")
        return 2
    glb = Path(pos[0]).resolve()
    if not glb.is_file():
        print("找不到", glb)
        return 2

    blender = next((p for p in BLENDER_CANDIDATES if p.is_file()), None)
    if blender is None:
        print("找不到 Blender，改 BLENDER_CANDIDATES")
        return 2

    suffix = "_flat" if flat else ""
    tmp = ROOT / ".tmp" / f"preview_{glb.stem}{suffix}"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [str(blender), "-b", "-P", str(ROOT / "tools" / "blender_preview.py"),
           "--", str(glb), str(tmp)]
    if flat:
        cmd.append("flat")
    subprocess.run(cmd, check=True)

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
    prefix = "FLAT " if flat else ""
    for i, im in enumerate(imgs):
        x = pad + (i % cols) * (w + pad)
        y = pad + (i // cols) * (h + head + pad)
        d.text((x + 6, y + 6), prefix + LABELS[i], fill=(230, 232, 236))
        sheet.paste(im, (x, y + head))
    out = ROOT / "out" / f"{glb.stem}{suffix}_views.png"
    sheet.save(out)
    print("预览:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
