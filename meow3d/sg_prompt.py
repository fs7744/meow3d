"""StableGen 的 `prompt_text_trellis2` 9 节点链，**API 格式** prompt 执行器。

模板参数逐项照 `D:/code/gen/others/StableGen-main/stablegen/util/workflow_templates.py:1642-1762`
与 `stablegen/mesh_gen/workflows.py:1258-1360` 的实际覆盖值：
- `tex_guidance_strength`：模板写 7.5，但运行时被
  `scene.trellis2_tex_guidance`（core/properties.py:1642，默认 **3.0**）覆盖；
- `texture_size`：`scene.trellis2_texture_size`（默认 **4096**）；
- `target_face_count`：`scene.trellis2_decimation`（默认 **100000**）；
- `seed`：`scene.trellis2_seed`（默认 0）。

与 meow3d/graph.py 的区别：那个执行 **UI 格式**（nodes + links + widgets_values），
这个执行 **API 格式**（{id: {class_type, inputs}}，连线写成 ["节点id", 槽位]），
正是 StableGen 发给 ComfyUI 的形式。

背景去除默认走 StableGen 的 **skip 分支**
（mesh_gen/workflows.py:1267-1277）：删掉 RemoveBackground 节点，
GetConditioning 直接连 LoadImage 的 image / mask。因为该节点要从 HF 拉
`ZhengPeng7/BiRefNet`；宿主侧已抠好的 RGBA 图经实测与它等价
（alpha 平均绝对差 0.0000）。用 `bg_removal='auto'` 可切回默认分支。
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import sg_io, sg_runtime

# 9 节点模板（StableGen prompt_text_trellis2）
PROMPT_TEXT_TRELLIS2: Dict[str, Dict[str, Any]] = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "", "upload": "image"}},
    # 模板写 precision="auto"，但 mesh_gen/workflows.py:1262-1264 会用
    # scene.trellis2_precision 覆盖，而 core/properties.py:1601-1610 的默认值是
    # **bf16**（不是 auto）。照实际运行值填 bf16。
    "2": {"class_type": "LoadTrellis2Models",
          "inputs": {"resolution": "1024_cascade", "precision": "bf16",
                     "attn_backend": "auto"}},
    "3": {"class_type": "Trellis2RemoveBackground",
          "inputs": {"image": ["1", 0], "low_vram": True}},
    "4": {"class_type": "Trellis2GetConditioning",
          "inputs": {"model_config": ["2", 0], "image": ["3", 0],
                     "mask": ["3", 1], "background_color": "black"}},
    "5": {"class_type": "Trellis2ImageToShape",
          "inputs": {"model_config": ["2", 0], "conditioning": ["4", 0],
                     "seed": 0, "ss_guidance_strength": 7.5,
                     "ss_sampling_steps": 12, "shape_guidance_strength": 7.5,
                     "shape_sampling_steps": 12, "max_tokens": 49152}},
    "6": {"class_type": "Trellis2ShapeToTexturedMesh",
          "inputs": {"model_config": ["2", 0], "conditioning": ["4", 0],
                     "shape_slat": ["5", 1], "subs": ["5", 2],
                     "seed": 0, "tex_guidance_strength": 3.0,
                     "tex_sampling_steps": 12}},
    "8": {"class_type": "Trellis2ProcessMesh",
          "inputs": {"trimesh": ["5", 0], "remesh": "on",
                     "remesh_band": 1.0, "remove_inner_faces": False,
                     "floater_threshold": 0.001, "target_face_count": 100000,
                     "weld_vertices": True, "weld_digits": 4,
                     "chart_cone_angle": 90.0, "chart_refine_iterations": 0,
                     "chart_global_iterations": 1, "chart_smooth_strength": 1}},
    "9": {"class_type": "Trellis2RasterizePBR",
          "inputs": {"trimesh": ["8", 0], "voxelgrid": ["6", 0],
                     "original_mesh": ["5", 0], "texture_size": 4096}},
    "7": {"class_type": "Trellis2ExportTrimesh",
          "inputs": {"trimesh": ["9", 0], "filename_prefix": "trellis2",
                     "file_format": "glb"}},
}

# `trellis2_skip_texture=True` 时的白模分支，照
# StableGen util/workflow_templates.py:1764-1852 `prompt_text_trellis2_shape_only`
# 与 mesh_gen/workflows.py:1231-1242 / 1316-1334 的运行时改写。
# 这是 `trellis2_texture_mode` 选 qwen_image_21 等扩散架构时的入口：
# callbacks.py:188 会置 skip_texture=True，mesh 阶段只出白模，
# 之后由 texturing 链（相机 → Qwen 逐视图 → 投影 → bake）上色。
PROMPT_TEXT_TRELLIS2_SHAPE_ONLY: Dict[str, Dict[str, Any]] = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "", "upload": "image"}},
    "2": {"class_type": "LoadTrellis2Models",
          "inputs": {"resolution": "1024_cascade", "precision": "bf16",
                     "attn_backend": "auto"}},
    "3": {"class_type": "Trellis2RemoveBackground",
          "inputs": {"image": ["1", 0], "low_vram": True}},
    "4": {"class_type": "Trellis2GetConditioning",
          "inputs": {"model_config": ["2", 0], "image": ["3", 0],
                     "mask": ["3", 1], "background_color": "black"}},
    "5": {"class_type": "Trellis2ImageToShape",
          "inputs": {"model_config": ["2", 0], "conditioning": ["4", 0],
                     "seed": 0, "ss_guidance_strength": 7.5,
                     "ss_sampling_steps": 12, "shape_guidance_strength": 7.5,
                     "shape_sampling_steps": 12, "max_tokens": 49152}},
    "6": {"class_type": "Trellis2Simplify",
          "inputs": {"trimesh": ["5", 0], "target_face_count": 100000,
                     "fill_holes": False, "remesh": True,
                     "remesh_band": 1.0}},
    "7": {"class_type": "Trellis2ExportTrimesh",
          "inputs": {"trimesh": ["6", 0], "filename_prefix": "trellis2",
                     "file_format": "glb"}},
}


class PromptError(RuntimeError):
    pass


def load_image(path: str | Path) -> Tuple[Any, Any]:
    """等价 ComfyUI LoadImage 的 [0]/[1]，但 mask 语义按 RemoveBackground 节点。

    节点包 `run_conditioning` (stages.py:821-825) 把 mask 直接当 alpha 用，
    而 `Trellis2RemoveBackground.execute` (nodes_inference.py:304-308) 输出的
    mask 就是 alpha/255。所以这里给 **前景=1** 的 mask。
    ComfyUI 内置 LoadImage 给的是 `1-alpha`（背景=1），两者不能混用。
    """
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGBA")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    image = torch.from_numpy(arr[..., :3].copy())[None]        # [1,H,W,3]
    mask = torch.from_numpy(arr[..., 3].copy())[None]          # [1,H,W]
    return image, mask


def build_prompt(*, image: str | Path, seed: int = 0, prefix: str = "trellis2",
                 bg_removal: str = "skip", texture_size: int = 4096,
                 target_face_count: int = 100000,
                 tex_guidance_strength: float = 3.0,
                 background_color: str = "black",
                 skip_texture: bool = False) -> Dict[str, Dict[str, Any]]:
    """按 StableGen 的运行时覆盖规则生成最终 prompt。

    `background_color` 是 `Trellis2GetConditioning` 原生选项
    （nodes_inference.py:35，black/gray/white），不是自创参数。
    `skip_texture=True` 对应 `scene.trellis2_skip_texture`
    （core/callbacks.py:188：texture_mode 非 native 时置 True），走
    `prompt_text_trellis2_shape_only` 出白模，供后续扩散贴图链上色。
    """
    import copy
    if background_color not in ("black", "gray", "white"):
        raise PromptError(f"background_color 只支持 black/gray/white，收到 {background_color!r}")
    p = copy.deepcopy(PROMPT_TEXT_TRELLIS2_SHAPE_ONLY if skip_texture
                      else PROMPT_TEXT_TRELLIS2)
    p["1"]["inputs"]["image"] = str(image)
    p["4"]["inputs"]["background_color"] = background_color
    prefix = f"{prefix}_{uuid.uuid4().hex[:8]}"
    p["7"]["inputs"]["filename_prefix"] = prefix
    p["5"]["inputs"]["seed"] = seed
    if skip_texture:
        # mesh_gen/workflows.py:1327-1330（use_pp=True + remesh_method='qdc'）
        p["6"]["inputs"]["target_face_count"] = target_face_count
        p["6"]["inputs"]["remesh"] = True
    else:
        p["6"]["inputs"]["seed"] = seed
        p["6"]["inputs"]["tex_guidance_strength"] = tex_guidance_strength
        p["9"]["inputs"]["texture_size"] = texture_size
        p["8"]["inputs"]["target_face_count"] = target_face_count

    if bg_removal == "skip":
        # StableGen mesh_gen/workflows.py:1274-1277
        del p["3"]
        p["4"]["inputs"]["image"] = ["1", 0]
        p["4"]["inputs"]["mask"] = ["1", 1]
    elif bg_removal != "auto":
        raise PromptError(f"bg_removal 只支持 skip/auto，收到 {bg_removal!r}")
    return p


def _expand_dynamic(inp: sg_io.Input, raw: Any, flat: Dict[str, Any]) -> Dict[str, Any]:
    """ComfyUI 前端把 DynamicCombo 的选项 + 子输入打包成 dict；这里等价展开。

    例：prompt 里 `"remesh": "on"` + 平铺的 `remesh_band` /
    `remove_inner_faces` → `{"remesh":"on", "remesh_band":1.0,
    "remove_inner_faces":False}`（Trellis2ProcessMesh.execute L334-340 读这个 dict）。
    """
    if not isinstance(raw, str):
        return raw
    out: Dict[str, Any] = {"remesh": raw}
    for opt in inp.options or []:
        if opt.name != raw:
            continue
        for sub in opt.inputs or []:
            if sub.default is not sg_io.MISSING:
                out[sub.name] = sub.default
    # prompt 里的平铺键优先
    for k, v in flat.items():
        if k in out or k in ("remesh_band", "remove_inner_faces",
                             "fill_holes", "fill_holes_perimeter"):
            out[k] = v
    return out


def _topo(prompt: Dict[str, Dict[str, Any]]) -> List[str]:
    deps: Dict[str, set] = {nid: set() for nid in prompt}
    for nid, node in prompt.items():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and str(v[0]) in prompt:
                deps[nid].add(str(v[0]))
    order: List[str] = []
    remaining = dict(deps)
    while remaining:
        ready = [n for n, d in remaining.items() if not d]
        if not ready:
            raise PromptError("prompt 有环或断线：" + ", ".join(remaining))
        for n in sorted(ready):
            order.append(n)
            del remaining[n]
        for d in remaining.values():
            d.difference_update(ready)
    return order


def execute(prompt: Dict[str, Dict[str, Any]], *,
            image: str | Path,
            on_node: Optional[Callable[[str, str, str], None]] = None
            ) -> Dict[str, Any]:
    """在当前进程按拓扑执行 9 节点，返回 {results, exports, elapsed}。"""
    boot = sg_runtime.bootstrap()
    mappings = boot["mappings"]
    results: Dict[str, List[Any]] = {}
    exports: List[str] = []

    img, mask = load_image(image)
    t0 = time.time()
    for nid in _topo(prompt):
        node = prompt[nid]
        ctype = node["class_type"]
        raw_inputs: Dict[str, Any] = dict(node.get("inputs") or {})
        if on_node:
            on_node(nid, ctype, "start")

        if ctype == "LoadImage":
            results[nid] = [img, mask]
            if on_node:
                on_node(nid, ctype, "done")
            continue

        cls = mappings.get(ctype)
        if cls is None:
            raise PromptError(f"节点未注册：{ctype} (id={nid})")
        schema = cls.define_schema()
        kwargs: Dict[str, Any] = {}
        for inp in schema.inputs:
            name = inp.name
            if name in raw_inputs:
                value = raw_inputs[name]
            elif inp.default is not sg_io.MISSING:
                value = inp.default
            elif inp.optional:
                continue
            else:
                continue
            # 连线
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                src, slot = str(value[0]), int(value[1])
                vals = results.get(src)
                if vals is None or slot >= len(vals):
                    raise PromptError(f"{ctype}.{name} 取不到 [{src}, {slot}]")
                value = vals[slot]
            if inp.dynamic:
                value = _expand_dynamic(inp, value, raw_inputs)
            kwargs[name] = value

        out = cls.execute(**kwargs)
        vals = list(out.args) if isinstance(out, sg_io.NodeOutput) else list(out)
        results[nid] = vals
        if ctype == "Trellis2ExportTrimesh" and vals:
            exports.append(str(vals[0]))
        if on_node:
            on_node(nid, ctype, "done")

    return {"results": results, "exports": exports,
            "elapsed": round(time.time() - t0, 1)}
