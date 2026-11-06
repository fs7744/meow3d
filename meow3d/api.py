"""对外就一个函数：`generate(image) -> .glb`。

官方 `MeshWithTexturing.json` 是调过参数的完整链路，我们只替换三样东西：
① 输入图片（不落 ComfyUI 的 input 目录，直接给 IMAGE tensor）；
② seed / 导出文件名；
③ 可选：贴图尺寸、目标面数。
其余一律照官方工作流执行，不自己重组节点。

## 与 StableGen 前半段的对齐（2026-11-05）

StableGen 的 image→mesh 用 `PozzettiAndrea/ComfyUI-TRELLIS2`
（installer.py:613，pin 9b878516）的 9 节点模板
`util/workflow_templates.py:1642-1762`；本项目的 `visualbruno` 节点包是另一套
实现，节点名/参数名不同，无法逐节点 1:1 复制。因此按**语义逐项对齐默认值**，
详见 `stablegen_align()` 与 README 的核对表。

一个关键事实：`Trellis2MeshTexturing` 节点自带默认就是
`texture_size=4096 / texture_guidance_strength=3.00`
（vendor/ComfyUI-Trellis2/nodes.py:2828,2832），**恰好等于 StableGen 默认值**；
是 `MeshWithTexturing.json` 这份示例把两者改成了 2048 / 5。所以对齐不需要换
节点包，只要把示例写死的值改回去，再补上 StableGen 默认开启的 BiRefNet 抠图。
"""
from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from . import graph, runtime

DEFAULT_WORKFLOW = (runtime.VENDOR / "ComfyUI-Trellis2" / "example_workflows"
                    / "MeshWithTexturing.json")

# StableGen 的 image→mesh 默认值（源码位置逐项标注）。
#   core/properties.py:1655  trellis2_texture_size = 4096
#   core/properties.py:1642  trellis2_tex_guidance = 3.0
#   core/properties.py:1660  trellis2_decimation  = 100000
#   util/workflow_templates.py:1723  remove_inner_faces = false
STABLEGEN_DEFAULTS: Dict[str, Any] = {
    "texture_size": 4096,
    "texture_guidance_strength": 3.0,
    "target_face_num": 100000,
    "remove_inner_faces": False,
}


def stablegen_align(wf: Dict[str, Any], *, enabled: bool = True) -> Dict[int, Dict[str, Any]]:
    """把官方示例工作流里被改低的参数改回 StableGen 默认值。

    ⚠️ **已证伪，默认路径不要调用**（`generate(align_stablegen=False)`）。
    2026-11-05 严格 A/B：同 `knight_matte.png` / seed 42 / `--max-size 2048`，
    只开这 5 项 → 四视角只剩几条细线，带纹理 GLB 25641 tris 全是退化薄片；
    对照组（官方原值）460365 tris、完整角色。根本原因是**参数语义错配**：
    StableGen 的 `target_face_count=100000` 是 `Trellis2ProcessMesh` 的**最终
    网格**，而这里 `target_face_num` 是 `SimplifyMesh` 的**中间网格**，
    `Trellis2MeshTexturing` 之后还会 xatlas 聚类再合并一次面；把两者当同一个
    数抄，等于连砍两次。另 `remove_inner_faces` 在 `ReconstructMeshWithQuad`
    里的语义也与 `Trellis2ProcessMesh` 不同。保留此函数仅作对照记录。

    只改**内存中**的 wf dict 与返回的 kwargs_overrides，不改 vendor 磁盘文件。
    返回 `graph.execute(kwargs_overrides=...)` 可直接消费的字典。

    对齐 5 项（其余 7 项本来就一致，不动）：
      1. BiRefNet 抠图：StableGen `Trellis2RemoveBackground` 默认开
         (workflow_templates.py:1665-1674, mesh_gen/workflows.py:1279)。
         这里由宿主侧 `image_tensor(bg_removal='auto')` 用同一个
         `birefnet-general-lite` 完成，工作流里保持 remove_background=False，
         避免二次抠图 —— 等价于 StableGen 的 'skip' 分支 + 客户端 rasterize。
      2. texture_size 2048 → 4096（PrimitiveInt 260）。
      3. texture_guidance_strength 5 → 3.0。
      4. target_face_num 500000 → 100000（PrimitiveInt 209）。
      5. remove_inner_faces true → false。
    """
    kw: Dict[int, Dict[str, Any]] = {}
    if not enabled:
        return kw
    for n in wf["nodes"]:
        t, nid = n["type"], int(n["id"])
        if t == "PrimitiveInt" and n.get("widgets_values"):
            v = n["widgets_values"][0]
            if v == 500000:                       # → 100000 面
                n["widgets_values"][0] = STABLEGEN_DEFAULTS["target_face_num"]
            elif v == 2048:                       # → 4096 图集
                n["widgets_values"][0] = STABLEGEN_DEFAULTS["texture_size"]
        elif t == "Trellis2MeshTexturing":
            kw[nid] = {
                "texture_size": STABLEGEN_DEFAULTS["texture_size"],
                "texture_guidance_strength": STABLEGEN_DEFAULTS["texture_guidance_strength"],
            }
        elif t == "Trellis2ReconstructMeshWithQuad":
            kw[nid] = {"remove_inner_faces": STABLEGEN_DEFAULTS["remove_inner_faces"]}
        elif t == "Trellis2PreProcessImage":
            kw[nid] = {"remove_background": False}   # 宿主侧已用 BiRefNet 抠好
    return kw


@lru_cache(maxsize=4)
def _matte_session(model: str):
    from rembg import new_session
    return new_session(model)


def image_tensor(path: str | Path, *, bg_removal: str = "auto",
                 matte_model: str = "birefnet-general-lite"):
    """读图 → ComfyUI IMAGE（**RGBA**，[1,H,W,4] float32 0~1）。

    必须是 4 通道：官方 `Trellis2PreProcessImage.preprocess_image` 无条件读
    `output_np[:, :, 3]` 当 alpha，用它算主体 bbox、裁成正方形，再把 RGB
    乘上 alpha 合成到黑底。给 3 通道会直接 IndexError。

    `bg_removal='auto'`（默认，对齐 StableGen 的 `Trellis2RemoveBackground`）
    先用 BiRefNet 抠出 alpha；`'skip'` 则沿用输入自带的 alpha，没有 alpha 时
    补一张全 255 的 alpha（等价于整图都是主体）。
    """
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    im = ImageOps.exif_transpose(Image.open(p))
    if bg_removal == "auto":
        from rembg import remove
        im = remove(im.convert("RGB"), session=_matte_session(matte_model))
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
             high_quality: bool = False, verbose: bool = True,
             align_stablegen: bool = False,
             bg_removal: str = "skip",
             matte_model: str = "birefnet-general-lite") -> Dict[str, Any]:
    """图 → 带原生纹理的 GLB。返回 {textured, white, exports, elapsed}。

    ⚠️ `align_stablegen` 默认 **False**：2026-11-05 严格 A/B 已证伪这组参数
    对齐（同输入/seed/max-size，只差这几项 → 模型退化成薄片）。详见 README。
    `bg_removal` 默认 skip = 沿用旧行为（输入需自带 alpha）。
    """
    boot = runtime.bootstrap()
    wf = graph.load_workflow(workflow)
    maps = boot["mappings"]

    prefix = prefix or f"meow3d/{Path(image).stem}_{seed}"
    kw: Dict[int, Dict[str, Any]] = _export_prefixes(wf, prefix)
    # 先按 StableGen 默认值对齐；后面的显式参数与 --hq 会覆盖它。
    for nid, params in stablegen_align(wf, enabled=align_stablegen).items():
        kw.setdefault(nid, {}).update(params)

    overrides: Dict[int, list] = {}
    img = image_tensor(image, bg_removal=bg_removal, matte_model=matte_model)
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
