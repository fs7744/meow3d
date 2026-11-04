"""一条命令跑完「引导投影」全流程：指引图 → Qwen 重绘 → 抠图 → 投影 bake。

    python tools/run_guided.py out/meow3d/asuka_Textured_00001_.glb \
        --reference out/in/asuka.png --cameras 4 --out out/meow3d/asuka_guided.glb

步骤（每一步都能单独跑，这里是编排）：

  1. `tools/guidance.py`        —— 同一台正交相机渲 depth/normal/clay/beauty
  2. `tools/qwen_guided_views.py` —— **用主项目 venv** 跑（py3.13），彩图 → 新视角
  3. `tools/matte.py`           —— 逐个抠成 RGBA（投影权重靠 alpha 排除背景）
  4. `tools/bake_projection.py` —— `--align similarity` + 遮挡剔除 + 二进制换贴图

## 为什么正面也要重绘（重要）

曾经把**原始 2D 图**直接投影到正面（"像素级直贴，最准"）。实测是错的：
TRELLIS 生成的 3D 网格比例和原图**不可能一致**，按包围盒线性映射之后，画面里的
刘海/眉毛会整体落到眼睛上 —— 渲染出来就是「两张脸叠在一起」的重影。

StableGen 不投影原图，它用网格渲的深度图**重新生成**正面，所以天然对齐。
这里跟它一致：原图只当「设计参考」，四个机位都由 Qwen 按网格渲的彩图重绘。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HOST_PY = Path(r"D:\code\2d-game-meow\.venv\Scripts\python.exe")
PY = Path(sys.executable)


def run(cmd, *, quiet=False):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    kw = dict(check=True)
    if quiet:
        kw.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([str(c) for c in cmd], **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", help="TRELLIS.2 输出的带贴图 GLB（白模也行）")
    ap.add_argument("--reference", required=True, help="原始 2D 参考图（正面）")
    ap.add_argument("--out", required=True, help="输出 GLB")
    ap.add_argument("--cameras", type=int, default=4, help="总机位数")
    ap.add_argument("--indices", default="0,1,2,3",
                    help="交给 Qwen 生成的机位。**包含 0（正面）**：正面的原图直贴会"
                         "因为网格比例不同而错位出重影，必须按网格重绘")
    ap.add_argument("--size", type=int, default=1024,
                    help="Qwen 出图边长。**1024 而不是 768**：768 的图贴进 2048 图集"
                         "后被放大 2~5 倍，脸和细节直接糊掉（实测对比）")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--exponent", type=float, default=8.0,
                    help="cos^k 权重指数；StableGen 生成后默认 15，3 会明显发糊")
    ap.add_argument("--detail", type=float, default=0.0)
    ap.add_argument("--work", help="中间目录，默认 .tmp/guided_<网格名>")
    ap.add_argument("--reuse-views", action="store_true",
                    help="复用 work 里已生成的视角（只重跑 bake）")
    args = ap.parse_args()

    mesh = Path(args.mesh).resolve()
    ref = Path(args.reference).resolve()
    if not mesh.is_file():
        print("找不到网格", mesh)
        return 2
    if not ref.is_file():
        print("找不到参考图", ref)
        return 2
    work = Path(args.work).resolve() if args.work else \
        ROOT / ".tmp" / f"guided_{mesh.stem}"
    guide = work / "guide"
    views = work / "views"
    ready = work / "ready"
    for d in (work, views, ready):
        d.mkdir(parents=True, exist_ok=True)
    idx = [int(x) for x in args.indices.split(",") if x.strip()]

    # --- 1. 指引图 ---
    print("[1/4] 渲指引图", flush=True)
    if not (guide / f"v{idx[0]:02d}_depth_c.png").is_file():
        run([PY, ROOT / "tools" / "guidance.py", mesh,
             "--cameras", args.cameras, "--indices", args.indices,
             "--out-dir", guide], quiet=True)
    else:
        print("  复用已有指引图", flush=True)

    # --- 2. Qwen 重绘 ---
    print("[2/4] Qwen 按指引重绘", flush=True)
    if not args.reuse_views or not all((views / f"view_{i:02d}.png").is_file()
                                       for i in idx):
        if not HOST_PY.is_file():
            print(f"缺主项目 venv：{HOST_PY}")
            return 2
        run([HOST_PY, ROOT / "tools" / "qwen_guided_views.py",
             "--ref", ref, "--guide", guide, "--mode", "beauty",
             "--cameras", args.cameras, "--indices", args.indices,
             "--size", args.size, "--steps", args.steps,
             "--out-dir", views])
    else:
        print("  复用已生成视角", flush=True)

    # --- 3. 抠图 ---
    print("[3/4] 抠图", flush=True)
    for i in idx:
        src = views / f"view_{i:02d}.png"
        dst = ready / f"view_{i:02d}.png"
        if not src.is_file():
            print(f"  缺 {src}")
            return 2
        if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        run([PY, ROOT / "tools" / "matte.py", src, dst], quiet=True)
        print(f"  view_{i:02d} 抠好", flush=True)

    # --- 4. 投影 bake ---
    print("[4/4] 投影 bake", flush=True)
    # **不给 --reference**：原图直接投影会因为网格比例不同而错位重影
    # （见模块 docstring）。四个机位全部来自「按网格重绘」的图。
    cmd = [PY, ROOT / "tools" / "bake_projection.py", mesh,
           "--views-dir", ready, "--cameras", args.cameras,
           "--view-indices", args.indices,
           "--align", "similarity", "--guide-dir", guide,
           "--exponent", args.exponent, "--detail", args.detail,
           "--out", args.out]
    run(cmd)
    print(f"\n完成：{args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
