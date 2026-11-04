"""给没有 UV 的白模展开 UV（投影贴图管线的前置步骤）。

    python tools/uv_unwrap.py 白模.glb 有UV.glb
    python tools/uv_unwrap.py 白模.glb 有UV.glb --decimate 100000

用 Blender 的 Smart UV Project。几十万面会很慢，建议先 --decimate。
输出 GLB 只带几何 + UV，不带材质，正好给 project_texture.py 当输入。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER_CANDIDATES = [
    Path(r"D:\code\gen\tools\blender\blender-4.5.9-windows-x64\blender.exe"),
    Path(r"D:\GAMES_tool\blender\blender.exe"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--decimate", type=int, default=0,
                    help="先降到这个面数再展开（0 = 不降面）")
    ap.add_argument("--blender")
    args = ap.parse_args()

    blender = Path(args.blender) if args.blender else next(
        (p for p in BLENDER_CANDIDATES if p.is_file()), None)
    if blender is None or not blender.is_file():
        print("找不到 Blender，用 --blender 指定路径")
        return 2
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not src.is_file():
        print("找不到", src)
        return 2
    dst.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([str(blender), "-b", "-P",
                    str(ROOT / "tools" / "blender_uv_unwrap.py"),
                    "--", str(src), str(dst), str(args.decimate)], check=True)
    print("完成:", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
