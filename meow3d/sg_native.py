"""StableGen v0.5.0 的 TRELLIS.2 链：ComfyUI 0.42+ **内置节点**，进程内执行。

节点序列照 Comfy-Org 的 TRELLIS.2 image-to-model 模板，与 StableGen
`mesh_gen/trellis2_graph.py`（v0.5.0）的事实结构一致：结构 SLat → 512 形状 →
级联上采样 → 形状解码 → 重网格/抽面 →（可选）纹理 SLat → UV 展开 →
从体素烘 PBR → 挂材质 → 导出 GLB。

与 v0.3.1 自定义节点链（`sg_prompt.py`）的区别：
- 不再需要 ComfyUI-TRELLIS2 自定义节点、comfy_env、comfy_sparse_attn、
  cumesh/flex_gemm/o_voxel 别名、drtk/nvdiffrast 替身；
- 权重换成 5 个单文件：`trellis_2_bf16` / `trellis_2_shape_vae_bf16` /
  `trellis_2_texture_vae_bf16` / `dino_v3_vit_l` / `birefnet`；
- 节点全部走 `comfy_api.latest._io.ComfyNode` 的 classmethod，可直接调用。

⚠️ 代码为独立实现，不复制 StableGen（GPL-3.0）源码。
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
COMFY = ROOT / "vendor" / "ComfyUI"

TRELLIS2_MODEL = "trellis_2_bf16.safetensors"
TRELLIS2_SHAPE_VAE = "trellis_2_shape_vae_bf16.safetensors"
TRELLIS2_TEXTURE_VAE = "trellis_2_texture_vae_bf16.safetensors"
TRELLIS2_IMAGE_ENCODER = "dino_v3_vit_l.safetensors"
TRELLIS2_BG_REMOVAL = "birefnet.safetensors"

BACKGROUND_COLORS = {"black": "#000000", "gray": "#808080", "white": "#FFFFFF"}
CASCADE_TARGETS = {"1024_cascade": 1024, "1536_cascade": 1536}

_BOOT: Optional[Dict[str, Any]] = None


class NativeError(RuntimeError):
    pass


def _install_server_stub() -> None:
    """`nodes_mesh_postprocess` 只为发进度 import `server`；给个最小替身，
    避免把 aiohttp / execution / app 整套拉进来。"""
    if "server" in sys.modules:
        return
    mod = types.ModuleType("server")

    class PromptServer:  # noqa: D401 - 只做 no-op
        instance = None

        def __init__(self, *a: Any, **k: Any):
            pass

        def send_sync(self, *a: Any, **k: Any):
            pass

        def send_progress_text(self, *a: Any, **k: Any):
            pass

        def send_progress(self, *a: Any, **k: Any):
            pass

        def queue_updated(self, *a: Any, **k: Any):
            pass

    PromptServer.instance = PromptServer()  # 节点会读 instance.xxx
    mod.PromptServer = PromptServer  # type: ignore[attr-defined]
    sys.modules["server"] = mod


def bootstrap() -> Dict[str, Any]:
    """注入替身、把 ComfyUI 加进 sys.path、收集节点类。幂等。"""
    global _BOOT
    if _BOOT is not None:
        return _BOOT
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    _install_server_stub()

    import importlib

    sources = {
        "nodes": ["UNETLoader", "VAELoader", "CLIPVisionLoader", "KSampler"],
        "comfy_extras.nodes_trellis2": [
            "Trellis2Conditioning", "Trellis2ShapeStage", "Trellis2UpsampleStage",
            "Trellis2TextureStage", "EmptyTrellis2LatentStructure",
            "VaeDecodeStructureTrellis2", "VaeDecodeShapeTrellis",
            "VaeDecodeTextureTrellis",
        ],
        "comfy_extras.nodes_mesh_postprocess": [
            "RemeshMesh", "DecimateMesh", "UnwrapMesh", "BakeTextureFromVoxel",
            "ApplyTextureToMesh",
        ],
        "comfy_extras.nodes_save_3d": ["SaveGLB"],
        "comfy_extras.nodes_bg_removal": ["LoadBackgroundRemovalModel", "RemoveBackground"],
        "comfy_extras.nodes_images": ["ImageCropToMask"],
        "comfy_extras.nodes_model_advanced": ["ModelSamplingSD3", "RescaleCFG"],
        "comfy_extras.nodes_custom_sampler": ["CFGOverride"],
        "comfy_extras.nodes_mask": ["InvertMask"],
    }
    mappings: Dict[str, Any] = {}
    missing: List[str] = []
    for modname, names in sources.items():
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{modname}: {type(exc).__name__}: {exc}")
            continue
        for n in names:
            cls = getattr(mod, n, None)
            if cls is None:
                missing.append(f"{modname}.{n}")
            else:
                mappings[n] = cls

    _BOOT = {"mappings": mappings, "missing": missing}
    return _BOOT


def configure_paths(*, models: str | Path | None = None,
                    output: str | Path | None = None) -> Dict[str, str]:
    """把 ComfyUI 的模型/输出目录指到本项目，避免往 vendor 里写文件。"""
    import folder_paths

    models_dir = Path(models) if models else (ROOT / "models")
    out_dir = Path(output) if output else (ROOT / "out")
    for name, sub in (("diffusion_models", "diffusion_models"),
                      ("vae", "vae"),
                      ("clip_vision", "clip_vision"),
                      ("background_removal", "background_removal")):
        folder_paths.folder_names_and_paths[name] = (
            [str(models_dir / sub)], {".safetensors", ".sft", ".gguf"})
    folder_paths.set_output_directory(str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"models": str(models_dir), "output": str(out_dir)}


def load_image(path: str | Path) -> Tuple[Any, Any]:
    """等价 ComfyUI `LoadImage`：IMAGE=[1,H,W,3] 0~1；MASK=[1,H,W]，
    **透明区域为 1**（所以后面要接 `InvertMask` 才是前景）。"""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    p = Path(path)
    if not p.is_file():
        raise NativeError(f"输入图不存在：{p}")
    im = ImageOps.exif_transpose(Image.open(p))
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(arr[..., :3])[None]
    mask = torch.from_numpy(1.0 - arr[..., 3])[None]
    return rgb, mask


def build_prompt(image: str | Path, *, resolution: str = "1024_cascade",
                 seed: int = 0, ss_steps: int = 12, ss_guidance: float = 7.5,
                 shape_steps: int = 12, shape_guidance: float = 7.5,
                 texture: bool = True, tex_steps: int = 12,
                 tex_guidance: float = 3.0, texture_size: int = 4096,
                 remove_background: bool = False, background: str = "black",
                 remesh: bool = True, target_faces: Optional[int] = 100000,
                 max_tokens: int = 49152, prefix: str = "trellis2",
                 ) -> Dict[str, Dict[str, Any]]:
    """生成 API 格式 prompt。默认参数取 StableGen v0.5.0 的场景默认值。"""
    prompt: Dict[str, Dict[str, Any]] = {}
    nodes: Dict[str, str] = {}

    def node(role: str, cls: str, **inputs: Any) -> str:
        nid = str(len(prompt) + 1)
        prompt[nid] = {"class_type": cls, "inputs": inputs}
        nodes[role] = nid
        return nid

    img = node("input_image", "LoadImage", image=str(image))
    unet = node("unet", "UNETLoader", unet_name=TRELLIS2_MODEL,
                weight_dtype="default")
    shape_vae = node("shape_vae", "VAELoader", vae_name=TRELLIS2_SHAPE_VAE)
    encoder = node("encoder", "CLIPVisionLoader", clip_name=TRELLIS2_IMAGE_ENCODER)

    if remove_background:
        bg_model = node("bg_model", "LoadBackgroundRemovalModel",
                        bg_removal_name=TRELLIS2_BG_REMOVAL)
        mask = [node("remove_bg", "RemoveBackground",
                     bg_removal_model=[bg_model, 0], image=[img, 0]), 0]
    else:
        # LoadImage 的 mask 是透明区；反相后前景=1。没有 alpha 的图整张都是主体。
        mask = [node("mask", "InvertMask", mask=[img, 1]), 0]

    crop = node("crop", "ImageCropToMask", images=[img, 0], masks=mask,
                width=1024, height=1024, pad_factor=1.0, grow_mask=0,
                background=BACKGROUND_COLORS.get(background, background))
    cond = node("conditioning", "Trellis2Conditioning",
                clip_vision_model=[encoder, 0], image=[crop, 0])

    # 结构 SLat：CFGOverride 区间 + RescaleCFG，再 ModelSamplingSD3 shift=5
    ss_interval = node("ss_interval", "CFGOverride", model=[unet, 0], cfg=1.0,
                       start_percent=0.667, end_percent=1.0)
    ss_rescale = node("ss_rescale", "RescaleCFG", multiplier=0.7,
                      model=[ss_interval, 0])
    ss_model = node("ss_model", "ModelSamplingSD3", shift=5.0,
                    model=[ss_rescale, 0])
    empty = node("empty", "EmptyTrellis2LatentStructure", batch_size=1)
    ss = node("ss_sampler", "KSampler", model=[ss_model, 0],
              positive=[cond, 0], negative=[cond, 1],
              latent_image=[empty, 0], seed=seed, steps=ss_steps,
              cfg=ss_guidance, sampler_name="euler", scheduler="normal",
              denoise=1.0)
    structure = node("structure", "VaeDecodeStructureTrellis2",
                     samples=[ss, 0], vae=[shape_vae, 0], resolution="32")

    # 512 形状
    shape_interval = node("shape_interval", "CFGOverride", model=[unet, 0],
                          cfg=1.0, start_percent=0.769, end_percent=1.0)
    shape_model = node("shape_rescale", "RescaleCFG", multiplier=0.5,
                       model=[shape_interval, 0])
    stage = node("shape_stage", "Trellis2ShapeStage", positive=[cond, 0],
                 negative=[cond, 1], voxel=[structure, 0])
    shape = node("shape_sampler", "KSampler", model=[shape_model, 0],
                 positive=[stage, 0], negative=[stage, 1],
                 latent_image=[stage, 2], seed=seed, steps=shape_steps,
                 cfg=shape_guidance, sampler_name="euler",
                 scheduler="normal", denoise=1.0)
    stage_out, shape_out = stage, shape

    # 级联上采样
    if resolution in CASCADE_TARGETS:
        up = node("upsample", "Trellis2UpsampleStage",
                  positive=[stage, 0], negative=[stage, 1],
                  shape_latent=[shape, 0], vae=[shape_vae, 0],
                  target_resolution=CASCADE_TARGETS[resolution])
        prompt[up]["inputs"]["max_tokens"] = max_tokens
        shape_out = node("hr_sampler", "KSampler", model=[shape_model, 0],
                         positive=[up, 0], negative=[up, 1],
                         latent_image=[up, 2], seed=seed, steps=shape_steps,
                         cfg=shape_guidance, sampler_name="euler",
                         scheduler="simple", denoise=1.0)
        stage_out = up

    decoded = node("shape", "VaeDecodeShapeTrellis", samples=[shape_out, 0],
                   vae=[shape_vae, 0])

    mesh = _remesh_and_decimate(node, [decoded, 0], remesh, target_faces)

    if texture:
        size = max(256, min(8192, int(texture_size) // 256 * 256))
        tex_vae = node("tex_vae", "VAELoader", vae_name=TRELLIS2_TEXTURE_VAE)
        tstage = node("tex_stage", "Trellis2TextureStage",
                      positive=[stage_out, 0], negative=[stage_out, 1],
                      shape_latent=[shape_out, 0])
        tex = node("tex_sampler", "KSampler", model=[unet, 0],
                   positive=[tstage, 0], negative=[tstage, 1],
                   latent_image=[tstage, 2], seed=seed, steps=tex_steps,
                   cfg=tex_guidance, sampler_name="euler",
                   scheduler="normal", denoise=1.0)
        voxels = node("texture", "VaeDecodeTextureTrellis", samples=[tex, 0],
                      vae=[tex_vae, 0], shape_subdivides=[decoded, 1])
        unwrapped = node("unwrap", "UnwrapMesh", mesh=mesh, segmenter="pec",
                         resolution=size, padding=1, weld_distance=0.0002)
        baked = node("bake", "BakeTextureFromVoxel", mesh=[unwrapped, 0],
                     voxel_colors=[voxels, 0], reference_mesh=[decoded, 0],
                     texture_size=size)
        mesh = [node("apply", "ApplyTextureToMesh", mesh=[unwrapped, 0],
                     base_color=[baked, 0], metallic=[baked, 1],
                     roughness=[baked, 2]), 0]

    node("export", "SaveGLB", mesh=mesh, filename_prefix=prefix)
    return prompt


def _remesh_and_decimate(node, mesh, remesh: bool, target_faces: Optional[int]):
    if remesh:
        # sign_mode 是 DynamicCombo：前端会展开成 dict，直调时得自己给。
        mesh = [node("remesh", "RemeshMesh", mesh=mesh, resolution=512,
                     sign_mode={"sign_mode": "udf"}, band=1.0,
                     project_back=0.0, fix_poles=False, smooth_iters=0,
                     drop_small_components=0.01,
                     precluster_max_verts=20000000), 0]
    if target_faces:
        mesh = [node("decimate", "DecimateMesh", mesh=mesh,
                     target_face_count=int(target_faces),
                     placement_mode={"placement_mode": "midpoint"}), 0]
    return mesh


def _topo(prompt: Dict[str, Dict[str, Any]]) -> List[str]:
    deps: Dict[str, set] = {nid: set() for nid in prompt}
    for nid, node in prompt.items():
        for v in (node.get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) \
                    and str(v[0]) in prompt:
                deps[nid].add(str(v[0]))
    order: List[str] = []
    remaining = dict(deps)
    while remaining:
        ready = [n for n, d in remaining.items() if not d]
        if not ready:
            raise NativeError("prompt 有环或断线：" + ", ".join(remaining))
        for n in sorted(ready, key=lambda x: int(x) if x.isdigit() else 0):
            order.append(n)
            del remaining[n]
        for d in remaining.values():
            d.difference_update(ready)
    return order


def _input_id(inp: Any) -> str:
    return getattr(inp, "id", None) or getattr(inp, "name", "")


def execute(prompt: Dict[str, Dict[str, Any]], *, image: str | Path,
            on_node: Optional[Callable[[str, str, str], None]] = None
            ) -> Dict[str, Any]:
    """按拓扑在进程内执行，`cls.execute(**kwargs)`。"""
    import torch

    boot = bootstrap()
    mappings = boot["mappings"]
    results: Dict[str, List[Any]] = {}
    exports: List[str] = []
    img, mask = load_image(image)
    t0 = time.time()

    # ComfyUI 的 execution.py:757 用 inference_mode 包住整条链；直调节点时
    # 少这层会让每个中间张量都带 grad_fn，显存和耗时都翻倍（2026-11-06 踩过）。
    with torch.inference_mode():
        for nid in _topo(prompt):
            node = prompt[nid]
            ctype = node["class_type"]
            raw: Dict[str, Any] = dict(node.get("inputs") or {})
            if on_node:
                on_node(nid, ctype, "start")
            if ctype == "LoadImage":
                results[nid] = [img, mask]
                if on_node:
                    on_node(nid, ctype, "done")
                continue

            cls = mappings.get(ctype)
            if cls is None:
                raise NativeError(
                    f"节点未注册：{ctype} (id={nid})"
                    + (f"；缺失：{boot['missing']}" if boot["missing"] else ""))
            _set_hidden(cls, nid, prompt)
            vals = _call_node(cls, raw, results)
            results[nid] = vals
            if ctype == "SaveGLB" and vals:
                exports.append(_as_path(vals[0]))
            if on_node:
                on_node(nid, ctype, "done")

    if not exports:
        # SaveGLB 只返回 `NodeOutput(ui=Saved3DModels(...))`，没有 outputs；
        # 按 filename_prefix 去输出目录找刚写的文件。
        exports = _guess_exports(prompt)
    return {"results": results, "exports": exports,
            "elapsed": round(time.time() - t0, 1)}


def _guess_exports(prompt: Dict[str, Dict[str, Any]]) -> List[str]:
    try:
        import folder_paths

        out_dir = Path(folder_paths.get_output_directory())
    except Exception:  # noqa: BLE001
        return []
    hits: List[str] = []
    for node in prompt.values():
        if node["class_type"] != "SaveGLB":
            continue
        pref = str(node["inputs"].get("filename_prefix") or "")
        if not pref:
            continue
        found = sorted(out_dir.glob(f"{pref}*.glb"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            hits.append(str(found[0]))
    return hits


def _resolve(value: Any, results: Dict[str, List[Any]], where: str) -> Any:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
        src, slot = str(value[0]), int(value[1])
        vals = results.get(src)
        if vals is None or slot >= len(vals):
            raise NativeError(f"{where} 取不到 [{src}, {slot}]")
        return vals[slot]
    return value


def _set_hidden(cls: Any, nid: str, prompt: Any) -> None:
    """ComfyUI 执行器会给节点注入 `cls.hidden`（execution.py:198-207）。
    `RemeshMesh` 等会读 `cls.hidden.unique_id` 发进度，没有就 AttributeError。"""
    try:
        from comfy_api.latest._io import HiddenHolder

        cls.hidden = HiddenHolder(unique_id=str(nid), prompt=prompt,
                                  extra_pnginfo=None, dynprompt=None,
                                  auth_token_comfy_org=None,
                                  api_key_comfy_org=None)
    except Exception:  # noqa: BLE001 - 旧式节点没有 hidden，忽略
        pass


def _call_node(cls: Any, raw: Dict[str, Any],
               results: Dict[str, List[Any]]) -> List[Any]:
    """兼容两种 ComfyUI 节点：
    - 新式 `IO.ComfyNode`：`define_schema()` + `cls.execute(**kwargs)`
    - 旧式：`INPUT_TYPES()` + `FUNCTION`，实例方法返回 tuple
    """
    if hasattr(cls, "define_schema"):
        schema = cls.define_schema()
        kwargs: Dict[str, Any] = {}
        for inp in schema.inputs:
            name = _input_id(inp)
            if name in raw:
                kwargs[name] = _resolve(raw[name], results, f"{cls.__name__}.{name}")
        out = cls.execute(**kwargs)
        return list(out.args) if hasattr(out, "args") else list(out)

    it = cls.INPUT_TYPES()
    kwargs = {}
    for section in ("required", "optional"):
        for name in (it.get(section) or {}):
            if name in raw:
                kwargs[name] = _resolve(raw[name], results, f"{cls.__name__}.{name}")
    fn = getattr(cls, "FUNCTION", None)
    if not fn:
        raise NativeError(f"{cls.__name__} 既非新式也无 FUNCTION")
    out = getattr(cls(), fn)(**kwargs)
    if isinstance(out, dict):           # 有些节点返回 {"ui":..., "result":(...)}
        out = out.get("result", ())
    if isinstance(out, tuple):
        return list(out)
    return [out]


def _as_path(v: Any) -> str:
    """SaveGLB 的返回值可能是路径字符串或 File3D 之类的对象。"""
    if isinstance(v, (str, Path)):
        return str(v)
    for attr in ("path", "filepath", "filename", "name"):
        got = getattr(v, attr, None)
        if isinstance(got, str):
            return got
    if isinstance(v, dict):
        for k in ("path", "filepath", "filename", "name"):
            if isinstance(v.get(k), str):
                return v[k]
    return str(v)
