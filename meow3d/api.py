"""对外就一个函数：`generate(image) -> .glb`。

官方 `MeshWithTexturing.json` 是调过参数的完整链路，我们只替换三样东西：
① 输入图片（不落 ComfyUI 的 input 目录，直接给 IMAGE tensor）；
② seed / 导出文件名；
③ 可选：贴图尺寸、目标面数。
其余一律照官方工作流执行，不自己重组节点。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import graph, runtime

DEFAULT_WORKFLOW = (runtime.VENDOR / "ComfyUI-Trellis2" / "example_workflows"
                    / "MeshWithTexturing.json")


def image_tensor(path: str | Path):
    """读图 → ComfyUI IMAGE（**RGBA**，[1,H,W,4] float32 0~1）。

    必须是 4 通道：官方 `Trellis2PreProcessImage.preprocess_image` 无条件读
    `output_np[:, :, 3]` 当 alpha，用它算主体 bbox、裁成正方形，再把 RGB
    乘上 alpha 合成到黑底。给 3 通道会直接 IndexError。
    没有 alpha 的图补一张全 255 的 alpha（等价于整图都是主体）。
    """
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    im = ImageOps.exif_transpose(Image.open(p))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
    else:
        rgba = im.convert("RGB")
        rgba.putalpha(255)
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]  # [1, H, W, 4]


def _export_prefixes(wf: Dict[str, Any], prefix: str) -> Dict[int, Dict[str, Any]]:
    """按官方工作流里 StringConcatenate 的后缀区分白模 / 贴图导出。"""
    links = graph._link_index(wf)
    byid = {int(n["id"]): n for n in wf["nodes"]}
    out: Dict[int, Dict[str, Any]] = {}
    for n in wf["nodes"]:
        if n["type"] != "Trellis2ExportMesh":
            continue
        suffix = ""
        for inp in n.get("inputs", []):
            if inp.get("name") != "filename_prefix" or inp.get("link") is None:
                continue
            l = links.get(int(inp["link"]))
            if not l:
                continue
            src = byid.get(int(l[1]), {})
            wv = " ".join(str(x) for x in (src.get("widgets_values") or []))
            if "_Textured" in wv:
                suffix = "_Textured"
            elif "_WhiteMesh" in wv:
                suffix = "_WhiteMesh"
        out[int(n["id"])] = {"filename_prefix": f"{prefix}{suffix}"}
    return out


def generate(image: str | Path, *, workflow: str | Path = DEFAULT_WORKFLOW,
             seed: int = 42, prefix: Optional[str] = None,
             texture_size: Optional[int] = None,
             target_faces: Optional[int] = None,
             max_size: Optional[int] = None,
             high_quality: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """图 → 带原生纹理的 GLB。返回 {textured, white, exports, elapsed}。"""
    boot = runtime.bootstrap()
    wf = graph.load_workflow(workflow)
    maps = boot["mappings"]

    prefix = prefix or f"meow3d/{Path(image).stem}_{seed}"
    kw: Dict[int, Dict[str, Any]] = _export_prefixes(wf, prefix)

    overrides: Dict[int, list] = {}
    img = image_tensor(image)
    for n in wf["nodes"]:
        nid, t = int(n["id"]), n["type"]
        if t == "Trellis2LoadImageWithTransparency":
            # 官方工作流连的是输出 #2 (image_with_alpha)
            overrides[nid] = [img, None, img]
        if t == "Trellis2LoadModel":
            # 全局注意力有内建 sdpa；稀疏注意力必须留 flash_attn
            #（windowed_attn 无兜底），由 runtime 的 shim 提供等价实现
            kw.setdefault(nid, {})["backend"] = "sdpa"
            kw.setdefault(nid, {})["sparse_backend"] = "flash_attn"
        if t in maps:
            slots = graph._widget_slots(maps[t])
            if "seed" in slots:
                kw.setdefault(nid, {})["seed"] = seed
            if "noise_seed" in slots:
                kw.setdefault(nid, {})["noise_seed"] = seed
            if texture_size is not None and "texture_size" in slots:
                kw.setdefault(nid, {})["texture_size"] = texture_size
            if target_faces is not None and "target_face_num" in slots:
                kw.setdefault(nid, {})["target_face_num"] = target_faces
        if max_size is not None and t == "Trellis2PreProcessImage":
            # 官方默认 1024 会把高清原图先缩掉；想让 2K 输入真正生效必须提上来
            kw.setdefault(nid, {})["max_size"] = max_size

    if high_quality:
        # 照抄官方 MeshTexturing_HighQuality.json：更多纹理步数、更强的引导、
        # 单视角（不让侧视渲染把正面五官平均掉）、DINO-Lock 让后期采样贴回原图。
        # 再把 UV 图集从 2048 提到 4096 —— 脸在图集里能多占一倍像素。
        for n in wf["nodes"]:
            if n["type"] == "Trellis2MeshTexturing":
                kw.setdefault(int(n["id"]), {}).update({
                    "texture_steps": 25,
                    "texture_guidance_strength": 7.5,
                    "texture_guidance_rescale": 0.01,
                    "texture_guidance_interval_end": 0.9,
                    "max_views": 1,
                    "dino_lock": 0.35,
                    "dino_substeps": 2,
                    "dino_foundation_cap": 0.92,
                    "texture_size": texture_size or 4096,
                })

    def log(nid, t, phase):
        if verbose and phase == "start":
            print(f"  → [{nid}] {t}", flush=True)

    t0 = time.time()
    res = graph.execute(wf, overrides=overrides, kwargs_overrides=kw, on_node=log)
    exports = [p for p in res["exports"] if p and p != "None"]

    out: Dict[str, Any] = {"exports": exports, "elapsed": round(time.time() - t0, 1),
                           "textured": None, "white": None}
    for p in exports:
        if "_Textured" in p:
            out["textured"] = p
        elif "_WhiteMesh" in p:
            out["white"] = p
    if out["textured"] is None and exports:
        out["textured"] = exports[-1]
    return out
