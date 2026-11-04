"""一条命令跑完「受保护的引导投影」全流程：指引图 → Qwen → 抠图 → 精修 → bake。

    python tools/run_guided.py out/meow3d/asuka_Textured_00001_.glb \
        --reference out/in/asuka.png --cameras 4 --out out/meow3d/asuka_guided.glb

步骤（每一步都能单独跑，这里是编排）：

  1. `tools/guidance.py`         —— 同机位渲 depth/normal/clay/beauty/**flat**
  2. `tools/qwen_guided_views.py`—— 用主项目 venv 跑，彩图 → 新视角
  3. `tools/matte.py`            —— 逐个抠成 RGBA
  4. `tools/refine_views.py`     —— **颜色锚定到 flat + 保护性局部合成**（本轮新增）
  5. `tools/bake_projection.py`  —— `--align frame` + 遮挡剔除 + 二进制换贴图

## 为什么加了第 4 步（与 StableGen 的真正差距）

StableGen 的 sequential 模式把 Qwen 输出当**补丁**：先渲一张「当前已投影
纹理」的 context render，未覆盖区涂品红，提示词明确要求保留非品红区域。
我们之前把 Qwen 输出当**整张新贴图**，于是 8 步 Lightning 的画风漂移被全盘
继承；而且起点是「已带贴图的 GLB」，整幅重绘本身就是净损失。

`refine_views.py` 用纯 numpy 复刻这个语义：以同机位 `flat`（Emission 直出，
等于贴图原色）为可信底图，先把 Qwen 结果颜色锚定回去，再按「底图局部清晰度」
决定放行多少像素。`--protect 0.85` 时平均只改写约 22% 的像素。

⚠️ 颜色基准**必须用 flat 不能用 beauty**：beauty 带灯光高光，写进贴图后
渲染时会被二次打光。实测用 beauty 当基准，贴图相对原生的平均 |ΔRGB| 只从
0.1755 降到 0.1670；换成 flat 后降到 **0.0816（-53.5%）**。

## 关于 --reference

`qwen_guided_views.py` 默认单图模式（`--two-images` 关闭）：只把模型自己
在该机位的彩图喂给 Qwen，参考图**不参与生成**——因为两张全身图一起进 Qwen
实测会画成「左原图 / 右结果」的两格拼图。参考图的颜色/设计信息由第 4 步的
flat 基准承载（flat 来自 GLB 原生贴图，颜色本来就是对的）。
需要把外部设计图强行注入时用 `--two-images`，但要先小范围验证。
"""
from __future__ import annotations

import argparse
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
                    help="Qwen 出图边长。1024 而不是 768：768 贴进 2048 图集会被放大")
    ap.add_argument("--steps", type=int, default=20,
                    help="采样步数。**默认 20**：对齐 StableGen QWEN 2.1 PRECISE "
                         "（20 步 / CFG1 / 无 Lightning）；8 步 Lightning 会把赛璐璐"
                         "重绘成半写实、颜色变淡、锐度掉一档")
    ap.add_argument("--lightning", action="store_true",
                    help="挂 8 步蒸馏 LoRA（快，但画质掉档；不推荐）")
    ap.add_argument("--lora", action="append", default=[],
                    help="额外 LoRA 名称:强度，可重复")
    ap.add_argument("--two-images", action="store_true",
                    help="把参考图一起喂给 Qwen（有拼图风险，默认关）")
    ap.add_argument("--exponent", type=float, default=8.0,
                    help="cos^k 权重指数；StableGen 生成后默认 15，3 会明显发糊")
    ap.add_argument("--protect", type=float, default=0.85,
                    help="保留 flat 底图的强度 0~1。带贴图 GLB 用 0.8~1；"
                         "白模（没有可信贴图）用 0")
    ap.add_argument("--anchor", type=float, default=1.0,
                    help="颜色锚定强度 0~1")
    ap.add_argument("--no-refine", action="store_true",
                    help="回到旧的「整幅重绘 + similarity 对齐」路径")
    ap.add_argument("--detail", type=float, default=0.0)
    ap.add_argument("--work", help="中间目录，默认 .tmp/guided_<网格名>")
    ap.add_argument("--reuse-views", action="store_true",
                    help="复用 work 里已生成的视角（只重跑精修与 bake）")
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
    refined = work / "refined"
    for d in (work, views, ready, refined):
        d.mkdir(parents=True, exist_ok=True)
    idx = [int(x) for x in args.indices.split(",") if x.strip()]

    # --- 1. 指引图（必须含 flat，颜色锚定的基准）---
    print("[1/5] 渲指引图", flush=True)
    guide_py = ROOT / "tools" / "guidance.py"
    gcmd = [PY, guide_py, mesh, "--cameras", args.cameras,
            "--indices", args.indices, "--out-dir", guide]
    if not (guide / f"v{idx[0]:02d}_depth_c.png").is_file():
        run([*gcmd, "--passes", "depth,normal,clay,beauty,flat"], quiet=True)
    elif any(not (guide / f"v{i:02d}_flat.png").is_file() for i in idx):
        print("  补渲 flat（颜色基准）", flush=True)
        run([*gcmd, "--passes", "flat"], quiet=True)
    else:
        print("  复用已有指引图", flush=True)

    # --- 2. Qwen 重绘 ---
    print(f"[2/5] Qwen 按指引重绘（{args.steps} 步"
          f"{'+Lightning' if args.lightning else '，无 Lightning'}）", flush=True)
    if not args.reuse_views or not all((views / f"view_{i:02d}.png").is_file()
                                       for i in idx):
        if not HOST_PY.is_file():
            print(f"缺主项目 venv：{HOST_PY}")
            return 2
        qcmd = [HOST_PY, ROOT / "tools" / "qwen_guided_views.py",
                "--ref", ref, "--guide", guide, "--mode", "beauty",
                "--cameras", args.cameras, "--indices", args.indices,
                "--size", args.size, "--steps", args.steps,
                "--out-dir", views]
        if not args.lightning:
            qcmd.append("--no-lightning")
        if args.two_images:
            qcmd.append("--two-images")
        for spec in args.lora:
            qcmd += ["--lora", spec]
        run(qcmd)
    else:
        print("  复用已生成视角", flush=True)

    # --- 3. 抠图 ---
    print("[3/5] 抠图", flush=True)
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

    # --- 4. 保护性精修 ---
    bake_views, align = ready, "similarity"
    if not args.no_refine:
        print(f"[4/5] 保护性精修（protect={args.protect}）", flush=True)
        run([PY, ROOT / "tools" / "refine_views.py", mesh,
             "--views", ready, "--guide", guide,
             "--cameras", args.cameras, "--indices", args.indices,
             "--out", refined, "--protect", args.protect,
             "--anchor", args.anchor])
        bake_views, align = refined, "frame"
    else:
        print("[4/5] 跳过精修（--no-refine，旧路径）", flush=True)

    # --- 5. 投影 bake ---
    print("[5/5] 投影 bake", flush=True)
    # **不给 --reference**：原图直接投影会因为网格比例不同而错位重影。
    cmd = [PY, ROOT / "tools" / "bake_projection.py", mesh,
           "--views-dir", bake_views, "--cameras", args.cameras,
           "--view-indices", args.indices,
           "--align", align, "--exponent", args.exponent,
           "--detail", args.detail, "--out", args.out,
           "--work-dir", work / "bake"]
    if align == "similarity":
        cmd += ["--guide-dir", guide]
    run(cmd)
    print(f"\n完成：{args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
