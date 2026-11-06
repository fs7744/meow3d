"""一张图 → 带 PBR 贴图的 GLB，跑 **StableGen v0.5.0** 的 TRELLIS.2 原生节点链。

    python tools/generate_native.py --plan
    python tools/generate_native.py "out/in/asuka_ref_matte.png" --seed 42
    python tools/generate_native.py "out/in/asuka_ref_matte.png" --no-texture

与 `tools/generate_sg.py` 的区别：那个跑 v0.3.1 的 ComfyUI-TRELLIS2 自定义
9 节点（需要 comfy_env / comfy_sparse_attn / cumesh 别名 / drtk 替身）；
这个跑 v0.5.0 的 **ComfyUI 0.42+ 内置节点**（UNETLoader / Trellis2Conditioning /
Trellis2ShapeStage / Trellis2UpsampleStage / Trellis2TextureStage /
UnwrapMesh / BakeTextureFromVoxel / SaveGLB …），进程内直接调用，
不启动 ComfyUI 服务。

权重（魔塔 `Comfy-Org/TRELLIS.2` + `Comfy-Org/BiRefNet`）：
    models/diffusion_models/trellis_2_bf16.safetensors
    models/vae/trellis_2_shape_vae_bf16.safetensors
    models/vae/trellis_2_texture_vae_bf16.safetensors
    models/clip_vision/dino_v3_vit_l.safetensors
    models/background_removal/birefnet.safetensors
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meow3d import sg_native  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="StableGen v0.5.0 TRELLIS.2 原生节点链（进程内）")
    ap.add_argument("image", nargs="?", help="输入图，建议已抠成 RGBA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="native")
    ap.add_argument("--resolution", choices=("512", "1024_cascade", "1536_cascade"),
                    default="1024_cascade")
    ap.add_argument("--target-faces", type=int, default=100000)
    ap.add_argument("--tex-guidance", type=float, default=3.0)
    ap.add_argument("--texture-size", type=int, default=4096)
    ap.add_argument("--tex-steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=49152,
                    help="级联上采样的体素预算；24G 卡上 49152 会 OOM，"
                         "降到 16384/8192 或改用 --resolution 512")
    ap.add_argument("--no-remesh", action="store_true")
    ap.add_argument("--no-texture", action="store_true",
                    help="只出白模（StableGen 的 texture_mode=none）")
    ap.add_argument("--remove-background", action="store_true",
                    help="跑 BiRefNet（需要 birefnet.safetensors）；默认用输入自带 alpha")
    ap.add_argument("--background", choices=("black", "gray", "white"), default="black")
    ap.add_argument("--plan", action="store_true", help="只打印 prompt，不加载模型")
    args = ap.parse_args()

    image = args.image or "out/in/asuka_ref_matte.png"
    prompt = sg_native.build_prompt(
        image=image, resolution=args.resolution, seed=args.seed,
        texture=not args.no_texture, tex_steps=args.tex_steps,
        tex_guidance=args.tex_guidance, texture_size=args.texture_size,
        remove_background=args.remove_background, background=args.background,
        remesh=not args.no_remesh, target_faces=args.target_faces,
        max_tokens=args.max_tokens, prefix=args.prefix,
    )

    if args.plan:
        boot = sg_native.bootstrap()
        print(f"节点 {len(prompt)} 个，已注册 {len(boot['mappings'])} 类"
              + (f"，缺失 {boot['missing']}" if boot["missing"] else ""))
        for nid in sg_native._topo(prompt):
            node = prompt[nid]
            print(f"[{nid}] {node['class_type']}")
            for k, v in node["inputs"].items():
                print(f"    {k} = {json.dumps(v, ensure_ascii=False)[:140]}")
        return 0

    if not Path(image).is_file():
        ap.error(f"找不到输入图：{image}")

    boot = sg_native.bootstrap()
    if boot["missing"]:
        ap.error("节点导入不完整：" + "; ".join(boot["missing"]))
    sg_native.configure_paths()

    def log(nid, ctype, phase):
        if phase == "start":
            print(f"  → [{nid}] {ctype}", flush=True)

    res = sg_native.execute(prompt, image=image, on_node=log)
    print(f"\n完成，用时 {res['elapsed']}s")
    for p in res["exports"]:
        print("  导出:", p)
    return 0 if res["exports"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
