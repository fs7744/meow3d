"""v0.5.0 的「白模 + Qwen 2.1 重贴」链：对应 `trellis2_texture_mode=qwen_image_21`。

StableGen v0.5.0 的 MESH + TEXTURE 预设**没有一个是 native**：
`QWEN 2.1 PRECISE/SAFE/ALT/VORONOI` 用 `qwen_image_21`，`DEFAULT/CHARACTERS/
ARCHITECTURE` 用 `sdxl`，只有 `MESH ONLY` 是 `none`。所以「官方成品效果」
= 白模 + 扩散重贴，本项目按这条接：

    1) tools/generate_native.py --no-texture        白模（v0.5.0 原生节点链）
    2) tools/run_guided.py --white-mesh             8 机位 → Qwen 2.1 → 投影 → bake

⚠️ 与 StableGen 的偏差：StableGen 在 Blender 里用逐面 BVH raycast 投影再
bake 到 BakeUV；本项目 `run_guided.py` 是 UV texel 反向映射。两者覆盖率
天花板不同（实测 native 投影 34%，见 README「投影贴图管线 v2」）。
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

# StableGen `QWEN 2.1 * (MESH + TEXTURE)` 预设的固定值（ui/presets.py:260-280）
PRESET = {"steps": 20, "cfg": 1.0, "sampler": "dpmpp_sde",
          "scheduler": "sgm_uniform", "lightning": False,
          "weight_exponent": 7.0, "cameras": 8}


def _run(cmd, *, capture_export=False):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=str(ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    exports = []
    for line in p.stdout:
        print("    " + line.rstrip(), flush=True)
        if capture_export and "导出:" in line:
            exports.append(line.split("导出:", 1)[1].strip())
    if p.wait():
        raise RuntimeError(f"命令失败：{cmd[1]}")
    return exports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", nargs="?", help="已抠好的 RGBA 图")
    ap.add_argument("--reference", default=None, help="2D 参考图，默认同输入图")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out/asuka_native_qwen21.glb")
    ap.add_argument("--prefix", default="native_q21")
    ap.add_argument("--cameras", type=int, default=PRESET["cameras"])
    ap.add_argument("--steps", type=int, default=PRESET["steps"])
    ap.add_argument("--exponent", type=float, default=PRESET["weight_exponent"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fill-frame", action="store_true")
    ap.add_argument("--reuse-white", default=None, help="复用已有白模 GLB")
    ap.add_argument("--work", default=None)
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    image = Path(args.image or "out/in/asuka_ref_matte.png").resolve()
    reference = Path(args.reference or image).resolve()
    out = Path(args.out).resolve()
    white = Path(args.reuse_white).resolve() if args.reuse_white else None

    mesh_cmd = [PY, ROOT / "tools" / "generate_native.py", image,
                "--seed", args.seed, "--resolution", "1024_cascade",
                "--no-texture", "--prefix", args.prefix + "_white"]
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
        print("StableGen v0.5.0：trellis2_texture_mode=qwen_image_21"
              "（MESH + TEXTURE 预设里没有 native）")
        print("\nQWEN 2.1 预设 → 本链传值：")
        for k, v in PRESET.items():
            local = {"steps": args.steps, "cameras": args.cameras,
                     "weight_exponent": args.exponent}.get(k, "Engine 默认")
            print(f"  {k:16} StableGen={str(v):14} 本地={local}")
        print("\n[1/2] 白模（v0.5.0 原生节点链，--no-texture）")
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

    if white is None:
        print("[1/2] v0.5.0 原生链出白模", flush=True)
        exports = _run(mesh_cmd, capture_export=True)
        if not exports:
            raise RuntimeError("白模阶段没有导出 GLB")
        white = Path(exports[-1])
        if not white.is_absolute():
            white = (ROOT / white).resolve()
        print(f"  白模：{white}", flush=True)
    else:
        print(f"[1/2] 复用白模：{white}", flush=True)

    guided_cmd[2] = white
    print("\n[2/2] Qwen 2.1 重贴", flush=True)
    _run(guided_cmd)

    print(f"\n完成，用时 {time.time() - t0:.1f}s")
    print(f"  白模: {white}")
    print(f"  成品: {out}"
          + (f"  ({out.stat().st_size / 1e6:.1f} MB)" if out.is_file() else "  !! 缺失"))
    return 0 if out.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
