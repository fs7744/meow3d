"""重建环境后的一键自检（不加载 15GB 权重、不占显存）。

    .venv/Scripts/python.exe tools/check_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ok = True

    def chk(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'OK' if cond else '!!'}] {label}{('  ' + detail) if detail else ''}")
        if not cond:
            ok = False

    print("Python:", sys.version.split()[0])
    import torch
    chk("torch", torch.__version__.startswith("2.8.0"), torch.__version__)
    chk("CUDA 构建", getattr(torch.version, "cuda", None) == "12.8",
        str(getattr(torch.version, "cuda", None)))
    chk("GPU 可用", torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    # custom_rasterizer 的 whl 只装了 custom_rasterizer_kernel.pyd，没有同名 py 模块；
    # UV 展开用的 xatlas 由节点内部按需调用，不作为顶层模块检查。
    for mod in ("cumesh", "o_voxel", "flex_gemm", "nvdiffrast",
                "nvdiffrec_render", "custom_rasterizer_kernel", "meshlib", "pymeshlab",
                "trimesh", "transformers", "rembg", "onnxruntime"):
        try:
            __import__(mod)
            chk(f"模块 {mod}", True)
        except Exception as e:  # noqa: BLE001
            chk(f"模块 {mod}", False, str(e)[:80])

    st = __import__("meow3d").runtime.available()
    chk("静态依赖", st["ok"], "; ".join(st["reasons"]))
    for p in (ROOT / "models/microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
              ROOT / "models/facebook/dinov3-vitl16-pretrain-lvd1689m/model.safetensors",
              ROOT / "models/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"):
        chk(p.relative_to(ROOT).as_posix(), p.is_file(),
            f"{p.stat().st_size / 2**20:.0f}MB" if p.is_file() else "缺失")

    try:
        boot = __import__("meow3d").runtime.bootstrap()
        chk("节点注册", len(boot["mappings"]) >= 80, f"{len(boot['mappings'])} 个")
        chk("models_dir", boot["folder_paths"].models_dir == str(ROOT / "models"),
            boot["folder_paths"].models_dir)
    except Exception as e:  # noqa: BLE001
        chk("运行时引导", False, str(e)[:160])

    print("\n结论:", "环境可用" if ok else "有缺项，按上面 [!!] 修")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
