"""StableGen 招牌链：图片 → TRELLIS.2 白模 → Qwen Image 2.1 多视图 → 投影贴图 → GLB。

对应 StableGen 的场景配置（不是自创流程）::

    trellis2_texture_mode = 'qwen_image_21'     # core/properties.py:1037-1041
      → trellis2_skip_texture = True            # core/callbacks.py:188
      → mesh 阶段跑 prompt_text_trellis2_shape_only（7 节点，白模）
      → texturing 阶段：相机绕模型 → 逐视图生成 → 投影混合 → bake
    preset = QWEN 2.1 PRECISE                   # ui/presets.py:82-105
      steps=20 / cfg=1.0 / dpmpp_sde / sgm_uniform / 无 Lightning
      weight_exponent=7.0 / qwen_image_21_native_rgba=True

README:22 那句 "views generated around it are projected into its texture"
指的就是本脚本接上的这一环；`tools/generate_sg.py` 默认只跑
`texture_mode='native'`（TRELLIS.2 自带贴图），不经过 Qwen。

用法::

    python tools/generate_sg_qwen21.py out/in/asuka_ref_matte.png \
        --reference out/in/asuka_ref.png --seed 42 --out out/asuka_qwen21.glb
    python tools/generate_sg_qwen21.py --plan        # 只打印对齐表与命令

⚠️ 与 StableGen 的偏差（如实标注，别当成「逐节点一模一样」）：

* StableGen 用 ComfyUI 的 `TextEncodeQwenImage21` + `QwenImage21Cache` 节点图
  （texturing/workflows.py:174-275）与 int8 单文件权重；这里用主项目
  `2d-game-meow` 的 diffusers Qwen-Image-2.1 Engine（权重已硬链接在
  `models/qwen/`），属于「做法对齐」，不是同一套节点。
* StableGen sequential 用「品红 context render + 保留非品红区域」的补丁式
  prompt；这里用 `refine_views.py` 的 flat 基准 + 保护性合成复刻该语义。
* 白模路线下 Qwen 输出经 `matte.py` 抠图，不是 Qwen 2.1 原生 RGBA。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PY = Path(sys.executable)

# StableGen `QWEN 2.1 PRECISE` 预设（ui/presets.py:82-105）
PRECISE = {
    "steps": 20,
    "cfg": 1.0,
    "sampler": "dpmpp_sde",
    "scheduler": "sgm_uniform",
    "lightning": False,
    "weight_exponent": 7.0,
    "camera_count": 8,
}


def _run(cmd, *, capture_export: bool = False):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1)
    exports = []
    for line in proc.stdout:
        print("    " + line.rstrip(), flush=True)
        if capture_export and "导出:" in line:
            exports.append(line.split("导出:", 1)[1].strip())
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"命令失败 rc={rc}: {cmd[1]}")
    return exports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="已抠成 RGBA 的输入图")
    ap.add_argument("--reference", default=None,
                    help="2D 参考原图（白模路线的颜色/设计来源），默认同输入图")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/sg_qwen21.glb")
    ap.add_argument("--prefix", default="sg_qwen21")
    ap.add_argument("--cameras", type=int, default=PRECISE["camera_count"])
    ap.add_argument("--steps", type=int, default=PRECISE["steps"])
    ap.add_argument("--exponent", type=float, default=PRECISE["weight_exponent"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fill-frame", action="store_true",
                    help="按模型投影跨度收紧画幅（头部有效像素 +30%）")
    ap.add_argument("--work", default=None, help="run_guided 中间目录")
    ap.add_argument("--reuse-white", default=None,
                    help="复用已有白模 GLB，跳过 mesh 阶段（调试用）")
    ap.add_argument("--plan", action="store_true", help="只打印对齐表与命令")
    args = ap.parse_args()

    image = Path(args.image or "out/in/asuka_ref_matte.png").resolve()
    reference = Path(args.reference or image).resolve()
    out = Path(args.out).resolve()
    white = Path(args.reuse_white).resolve() if args.reuse_white else None

    mesh_cmd = [PY, ROOT / "tools" / "generate_sg.py", image,
                "--seed", args.seed, "--shape-only",
                "--prefix", args.prefix + "_shape"]
    guided_cmd = [PY, ROOT / "tools" / "run_guided.py", white or Path("<白模.glb>"),
                  "--reference", reference, "--out", out,
                  "--white-mesh", "--auto-cameras", args.cameras,
                  "--steps", args.steps, "--size", args.size,
                  "--exponent", args.exponent, "--decimate", 0]
    if args.fill_frame:
        guided_cmd.append("--fill-frame")
    if args.work:
        guided_cmd += ["--work", args.work]

    if args.plan:
        print("StableGen 场景配置：trellis2_texture_mode=qwen_image_21 / "
              "trellis2_skip_texture=True / preset=QWEN 2.1 PRECISE")
        print("\n预设参数（ui/presets.py:82-105）→ 本链传值：")
        for k, v in PRECISE.items():
            local = {"steps": args.steps, "camera_count": args.cameras,
                     "weight_exponent": args.exponent}.get(k, "Engine 默认")
            print(f"  {k:20} StableGen={str(v):14} 本地={local}")
        print("\n[1/2] 白模（prompt_text_trellis2_shape_only）")
        print("  $ " + " ".join(str(c) for c in mesh_cmd))
        print("\n[2/2] Qwen 2.1 多视图 + 投影 bake")
        print("  $ " + " ".join(str(c) for c in guided_cmd))
        return 0

    if not image.is_file():
        ap.error(f"找不到输入图：{image}")
    if not reference.is_file():
        ap.error(f"找不到参考图：{reference}")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # --- 1/2 白模 ---
    if white is None:
        print("[1/2] TRELLIS.2 白模（StableGen shape_only 7 节点）", flush=True)
        exports = _run(mesh_cmd, capture_export=True)
        if not exports:
            raise RuntimeError("mesh 阶段没有导出 GLB")
        white = Path(exports[-1])
        if not white.is_absolute():
            white = (ROOT / white).resolve()
        print(f"  白模：{white}", flush=True)
    else:
        print(f"[1/2] 复用白模：{white}", flush=True)

    guided_cmd[2] = white
    print("\n[2/2] Qwen Image 2.1 多视图 → 投影 bake", flush=True)
    _run(guided_cmd)

    print(f"\n完成，用时 {time.time() - t0:.1f}s")
    print(f"  白模: {white}")
    print(f"  成品: {out}  ({out.stat().st_size / 1e6:.1f} MB)"
          if out.is_file() else f"  !! 未找到成品 {out}")
    return 0 if out.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
