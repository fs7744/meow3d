"""让 StableGen 指定的 `PozzettiAndrea/ComfyUI-TRELLIS2`（pin 9b878516）
在本项目**当前进程**里跑起来，不启动 ComfyUI 服务、不改 vendor 一行。

与 meow3d/runtime.py 的区别：那条链是 visualbruno 的 UI 格式工作流，这条链
是 StableGen 的 **API 格式 prompt**（`prompt_text_trellis2` 9 节点）。两者对
`comfy_api.latest` 的替身不同，因此各自独立 bootstrap，不要在同一进程混用。

替身边界（源码核对结论）：
- 纯描述层：`comfy_api.latest.io` → meow3d/sg_io.py
- ComfyUI 运行时：`comfy.model_management` / `comfy.utils` /
  `comfy.model_patcher` / `comfy_env` / `comfy.sparse*` → 轻量等价实现
- 真计算依赖全部复用本机已装的编译轮子，只做**模块别名**：
  cumesh_vb→cumesh / flex_gemm_ap→flex_gemm / o_voxel_vb_ap→o_voxel
- `drtk`（pixi 渠道包，PyPI 同名包是占位）→ meow3d/sg_drtk.py 等价实现
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
COMFY_DIR = VENDOR / "ComfyUI"
SG_DIR = VENDOR / "_sg" / "ComfyUI-TRELLIS2-9b878516f2dc2fd873f4f6cceadba403dd12d83e"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "out"
INPUT_DIR = ROOT / "in"
TMP_DIR = ROOT / ".tmp"

_boot: Optional[Dict[str, Any]] = None


class SGRuntimeError(RuntimeError):
    pass


def available() -> Dict[str, Any]:
    reasons = []
    for p, what in ((COMFY_DIR, "ComfyUI 运行时"), (SG_DIR, "StableGen 指定节点包")):
        if not p.is_dir():
            reasons.append(f"缺 {what}：{p}")
    if not (MODELS_DIR / "trellis2" / "pipeline.json").is_file():
        reasons.append(f"缺 models/trellis2/pipeline.json（先跑 tools/link_sg_models.py）")
    return {"ok": not reasons, "reasons": reasons, "sg_dir": str(SG_DIR)}


# --------------------------------------------------------------------------
# 描述层 / ComfyUI 运行时替身
# --------------------------------------------------------------------------

def _install_io_stub() -> None:
    from . import sg_io

    pkg = ModuleType("comfy_api")
    latest = ModuleType("comfy_api.latest")
    latest.io = sg_io  # type: ignore[attr-defined]

    class File3D:
        def __init__(self, path: str):
            self.path = path

        def __str__(self) -> str:
            return self.path

    latest.Types = type("Types", (), {"File3D": File3D})  # type: ignore[attr-defined]
    pkg.latest = latest  # type: ignore[attr-defined]
    sys.modules["comfy_api"] = pkg
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.io"] = sg_io


def _install_kitchen_stub() -> None:
    """`comfy_kitchen` 只服务 fp8/fp4/int8 路径，本链全用 bf16。

    vendor/ComfyUI 的 comfy.ldm.modules.attention / model_base /
    supported_models_base 在 **import 阶段**就会读
    `comfy_kitchen.int8_attention_is_available` 等符号，缺了直接 AttributeError。
    给一个「任何属性都是返回 False 的 no-op」模块即可，不参与计算。
    """

    class _Any:
        """既能被当函数调用、又能继续取属性的惰性 no-op。"""

        def __call__(self, *_a: Any, **_k: Any):
            return _Result()

        def __getattr__(self, _name: str):
            return _Any()

        def __bool__(self):
            return False

    class _Result(_Any):
        """探测型调用的返回值：falsy、可迭代、可 .items()。

        comfy/quant_ops.py 顶层会 `ck.list_backends().items()`，
        attention 模块会 `if ck.int8_attention_is_available():`。
        """

        def items(self):
            return []

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

        def __contains__(self, _x: Any):
            return False

        def __getitem__(self, _k: Any):
            return _Any()

    class _Kitchen(ModuleType):
        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Any()

    kitchen = _Kitchen("comfy_kitchen")
    kitchen.__path__ = []  # type: ignore[attr-defined]
    sys.modules["comfy_kitchen"] = kitchen
    for name in ("tensor", "int8_decode", "fp8", "fp4"):
        sub = _Kitchen(f"comfy_kitchen.{name}")
        sys.modules[f"comfy_kitchen.{name}"] = sub
        setattr(kitchen, name, sub)


def _install_comfy_env_stub() -> None:
    env = ModuleType("comfy_env")
    env.register_nodes = lambda *a, **k: ({}, {})  # type: ignore[attr-defined]
    env.setup_env = lambda *a, **k: None  # type: ignore[attr-defined]
    env.copy_files = lambda *a, **k: None  # type: ignore[attr-defined]
    env.install = lambda *a, **k: None  # type: ignore[attr-defined]
    sys.modules["comfy_env"] = env


def _install_sparse_stub() -> None:
    """comfy_sparse_attn 的三个模块 → comfy.sparse / ops_sparse / attention_sparse。

    节点包原本靠 `setup_link()` 往 ComfyUI/comfy/ 建软链接；这里改为直接注册
    sys.modules，避免改动 vendor 目录。
    """
    import comfy_sparse_attn

    # 装配函数改 no-op（否则会写 vendor/ComfyUI/comfy/）
    comfy_sparse_attn.setup_link = lambda *a, **k: None  # type: ignore[attr-defined]

    import comfy
    for short, mod_name in (("sparse", "sparse"),
                            ("ops_sparse", "ops_sparse"),
                            ("attention_sparse", "attention_sparse")):
        m = importlib.import_module(f"comfy_sparse_attn.{mod_name}")
        sys.modules[f"comfy.{short}"] = m
        setattr(comfy, short, m)


def _install_mm_stub() -> None:
    import torch

    def _false(*_a: Any, **_k: Any):
        return False

    class _Tolerant(ModuleType):
        """ComfyUI 各模块会探测大量 mm/utils 属性（is_amd / is_intel_xpu /
        vae_dtype / ...）；未知的给一个返回 False 的 no-op，避免逐个补。"""

        def __getattr__(self, name: str):
            if name.startswith("__"):
                raise AttributeError(name)
            return _false

    mm = _Tolerant("comfy.model_management")
    mm.current_loaded_models = []  # type: ignore[attr-defined]

    def device():
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    mm.get_torch_device = lambda *a, **k: device()  # type: ignore[attr-defined]
    mm.should_use_bf16 = lambda *a, **k: True  # type: ignore[attr-defined]
    mm.should_use_fp16 = lambda *a, **k: False  # type: ignore[attr-defined]
    mm.get_total_memory = lambda *a, **k: (  # type: ignore[attr-defined]
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)
    mm.free_memory = lambda *a, **k: (  # type: ignore[attr-defined]
        torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0)
    mm.soft_empty_cache = lambda *a, **k: (  # type: ignore[attr-defined]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None)
    mm.throw_exception_if_processing_interrupted = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.unload_all_models = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.intermediate_device = lambda *a, **k: torch.device("cpu")  # type: ignore[attr-defined]
    mm.unet_offload_device = lambda *a, **k: torch.device("cpu")  # type: ignore[attr-defined]
    mm.vae_dtype = lambda *a, **k: torch.bfloat16  # type: ignore[attr-defined]
    mm.unet_dtype = lambda *a, **k: torch.bfloat16  # type: ignore[attr-defined]
    mm.is_device_cpu = lambda d, *a, **k: getattr(d, "type", "cpu") == "cpu"  # type: ignore[attr-defined]

    # comfy/ops.py 的 CastBiasWeightContext 会真实调用这些张量搬运函数；
    # 兜底返回 False 会让 weight/bias 变成 bool（“'bool' object has no
    # attribute 'to'”）。这里给等价实现，并让 offload 路径整体关闭。
    def cast_to(tensor, dtype=None, device=None, non_blocking=False,
                copy=False, stream=None, r=None):
        if r is not None:
            return r
        if tensor is None:
            return None
        if device is not None:
            tensor = tensor.to(device, non_blocking=non_blocking)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor

    mm.cast_to = cast_to  # type: ignore[attr-defined]
    mm.cast_to_device = lambda tensor, device=None, dtype=None, **k: cast_to(  # type: ignore[attr-defined]
        tensor, dtype=dtype, device=device, **{x: k[x] for x in ("non_blocking",) if x in k})
    mm.cast_to_gathered = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.get_offload_stream = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.get_cast_buffer = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.get_aimdo_cast_buffer = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.sync_stream = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.current_stream = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.device_supports_non_blocking = lambda *a, **k: False  # type: ignore[attr-defined]
    mm.lora_compute_dtype = lambda *a, **k: torch.float32  # type: ignore[attr-defined]
    mm.unet_manual_cast = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.LARGEST_AIMDO_CASTED_WEIGHT = 0  # type: ignore[attr-defined]
    mm.torch_version_numeric = (2, 8, 0)  # type: ignore[attr-defined]
    mm.OOM_EXCEPTION = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)  # type: ignore[attr-defined]
    for flag in ("in_training", "training_fp8_bwd", "supports_fp8_compute",
                 "supports_int8_compute", "supports_mxfp8_compute",
                 "supports_nvfp4_compute"):
        setattr(mm, flag, False)

    # comfy/ldm/modules/attention.py 的后端选择；
    # force_upcast_attention_dtype 必须返回 None（返回 False 会让
    # `current_dtype in FORCE_UPCAST_ATTENTION_DTYPE` 抛 TypeError）。
    mm.force_upcast_attention_dtype = lambda *a, **k: None  # type: ignore[attr-defined]
    mm.get_free_memory = lambda *a, **k: (  # type: ignore[attr-defined]
        torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0)
    mm.pytorch_attention_enabled = lambda *a, **k: True  # type: ignore[attr-defined]
    mm.is_nvidia = lambda *a, **k: True  # type: ignore[attr-defined]
    mm.raise_non_oom = lambda *a, **k: None  # type: ignore[attr-defined]
    for flag in ("xformers_enabled", "flash_attention_enabled",
                 "sage_attention_enabled", "comfy_kitchen_attention_enabled"):
        setattr(mm, flag, lambda *a, **k: False)

    def load_models_gpu(models, *a, **k):
        for patcher in models or []:
            model = getattr(patcher, "model", None)
            target = getattr(patcher, "load_device", None) or device()
            if model is not None and hasattr(model, "to"):
                try:
                    model.to(target)
                except Exception:  # noqa: BLE001
                    pass
        return 0

    mm.load_models_gpu = load_models_gpu  # type: ignore[attr-defined]
    sys.modules["comfy.model_management"] = mm
    setattr(sys.modules["comfy"], "model_management", mm)

    utils = _Tolerant("comfy.utils")

    class ProgressBar:
        def __init__(self, total=None, *a, **k):
            self.total, self.current = total, 0

        def update(self, value=1, *a, **k):
            self.current += value

        def update_absolute(self, value, total=None, *a, **k):
            self.current = value
            if total is not None:
                self.total = total

        def __getattr__(self, _name):
            return lambda *a, **k: None

    def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
        from safetensors.torch import load_file
        sd = load_file(ckpt, device=str(device) if device is not None else "cpu")
        return (sd, {}) if return_metadata else sd

    def common_upscale(samples, width, height, upscale_method="bicubic", crop="disabled"):
        x = samples.movedim(-1, 1)
        mode = {"nearest": "nearest", "bilinear": "bilinear",
                "bicubic": "bicubic", "area": "area"}.get(upscale_method, "bicubic")
        align = None if mode in ("nearest", "area") else False
        y = torch.nn.functional.interpolate(x, size=(height, width), mode=mode,
                                            align_corners=align)
        return y.movedim(1, -1)

    utils.ProgressBar = ProgressBar  # type: ignore[attr-defined]
    utils.load_torch_file = load_torch_file  # type: ignore[attr-defined]
    utils.common_upscale = common_upscale  # type: ignore[attr-defined]
    sys.modules["comfy.utils"] = utils
    setattr(sys.modules["comfy"], "utils", utils)

    patcher = _Tolerant("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, model, load_device=None, offload_device=None, **kw):
            self.model = model
            self.load_device = load_device or torch.device("cpu")
            self.offload_device = offload_device or torch.device("cpu")

        def unpatch_model(self, device_to=None, *a, **k):
            if device_to is not None and hasattr(self.model, "to"):
                try:
                    self.model.to(device_to)
                except Exception:  # noqa: BLE001
                    pass

        def __getattr__(self, _name):
            return lambda *a, **k: None

    patcher.ModelPatcher = ModelPatcher  # type: ignore[attr-defined]
    sys.modules["comfy.model_patcher"] = patcher
    setattr(sys.modules["comfy"], "model_patcher", patcher)


def _install_wheel_aliases() -> None:
    """节点包用的 `_vb` / `_ap` 后缀轮子，与本机已装轮子 API 一致，做别名即可。"""
    cumesh = importlib.import_module("cumesh")
    flex = importlib.import_module("flex_gemm")
    ovox = importlib.import_module("o_voxel")

    alias = {
        "cumesh_vb": cumesh,
        "flex_gemm_ap": flex,
        "o_voxel_vb_ap": ovox,
    }
    for dotted in ("flex_gemm.ops",
                   "flex_gemm.ops.grid_sample",
                   "flex_gemm.ops.spconv",
                   "flex_gemm.ops.spconv.submanifold_conv3d",
                   "o_voxel.convert"):
        try:
            alias["flex_gemm_ap." + dotted[len("flex_gemm."):] if dotted.startswith("flex_gemm.")
                  else "o_voxel_vb_ap.convert"] = importlib.import_module(dotted)
        except Exception:  # noqa: BLE001
            pass
    for name, mod in alias.items():
        sys.modules[name] = mod

    # `flex_gemm_ap` 比本机 `flex_gemm` 多一个 SubMConv3dFunction.tiled_forward
    # （comfy_sparse_attn/ops_sparse.py:235 在低显存 + 大 voxel 数时调用）。
    # 它只是分块省显存，用本机已有的普通前向往上补一个等价实现即可。
    try:
        conv = importlib.import_module("flex_gemm.ops.spconv.submanifold_conv3d")
        fn = conv.SubMConv3dFunction
        if not hasattr(fn, "tiled_forward"):
            def tiled_forward(feats_cpu, coords, shape, weight, bias,
                              kernel_size, dilation, max_voxels_per_tile=0):
                feats = feats_cpu.to(weight.device)
                cache = fn._compute_neighbor_cache(
                    coords, shape, tuple(kernel_size), dilation)
                return fn._sparse_submanifold_conv_forward(
                    feats, cache, weight, bias)

            fn.tiled_forward = staticmethod(tiled_forward)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def _install_drtk_stub() -> None:
    from . import sg_drtk
    sys.modules["drtk"] = sg_drtk


# --------------------------------------------------------------------------
# 装载节点包（跳过两层装配 __init__.py，避免 setup_link 改 vendor）
# --------------------------------------------------------------------------

def _load_sg_nodes() -> Dict[str, Any]:
    pkg = ModuleType("sg_trellis2")
    pkg.__path__ = [str(SG_DIR)]  # type: ignore[attr-defined]
    sys.modules["sg_trellis2"] = pkg

    sub = ModuleType("sg_trellis2.nodes")
    sub.__path__ = [str(SG_DIR / "nodes")]  # type: ignore[attr-defined]
    sys.modules["sg_trellis2.nodes"] = sub
    pkg.nodes = sub  # type: ignore[attr-defined]

    mappings: Dict[str, Any] = {}
    for name in ("nodes_loader", "nodes_inference", "nodes_unwrap"):
        mod = importlib.import_module(f"sg_trellis2.nodes.{name}")
        mappings.update(getattr(mod, "NODE_CLASS_MAPPINGS", {}))
    return mappings


def bootstrap() -> Dict[str, Any]:
    """幂等。→ {mappings, folder_paths, ...}"""
    global _boot
    if _boot is not None:
        return _boot

    for d in (OUT_DIR, INPUT_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    for p in (str(COMFY_DIR), str(VENDOR), str(SG_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 路径覆盖必须在 import folder_paths 之前
    import comfy.cli_args as cli_args
    cli_args.args.base_directory = str(ROOT)
    cli_args.args.models_directory = str(MODELS_DIR)
    cli_args.args.output_directory = str(OUT_DIR)
    cli_args.args.input_directory = str(INPUT_DIR)
    cli_args.args.temp_directory = str(TMP_DIR)

    server = ModuleType("server")

    class _StubServer:
        instance = None

        def send_sync(self, *a, **k):
            pass

        def send(self, *a, **k):
            pass

    _StubServer.instance = _StubServer()
    server.PromptServer = _StubServer  # type: ignore[attr-defined]
    sys.modules.setdefault("server", server)

    _install_io_stub()
    # 故意不装 comfy_kitchen：comfy/quant_ops.py:7-45 有
    # `except ImportError` 的 fp8 降级分支，缺包会走 fallback；硬造 no-op
    # 替身反而会让 `class _TensorCoreFP8LayoutBase(_CKFp8Layout)` 炸掉。
    _install_comfy_env_stub()
    # mm 必须早于 sparse：comfy_sparse_attn.ops_sparse 会 `import comfy.ops`，
    # 而 comfy.ops 会 `import comfy.model_management`；晚装就拖进真的
    # model_management → memory_management → quant_ops 整条链。
    _install_mm_stub()
    _install_sparse_stub()
    _install_wheel_aliases()
    _install_drtk_stub()

    import folder_paths
    folder_paths.models_dir = str(MODELS_DIR)
    folder_paths.output_directory = str(OUT_DIR)
    folder_paths.input_directory = str(INPUT_DIR)
    folder_paths.temp_directory = str(TMP_DIR)
    folder_paths.base_path = str(ROOT)

    try:
        mappings = _load_sg_nodes()
    except Exception as exc:  # pragma: no cover - 环境诊断
        raise SGRuntimeError(f"装载 StableGen 节点包失败：{exc}") from exc

    required = ("LoadTrellis2Models", "Trellis2RemoveBackground",
                "Trellis2GetConditioning", "Trellis2ImageToShape",
                "Trellis2ShapeToTexturedMesh", "Trellis2ProcessMesh",
                "Trellis2RasterizePBR", "Trellis2ExportTrimesh")
    missing = [n for n in required if n not in mappings]
    if missing:
        raise SGRuntimeError("节点未注册：" + "、".join(missing))

    _boot = {"mappings": mappings, "folder_paths": folder_paths,
             "output": OUT_DIR, "models": MODELS_DIR, "sg_dir": SG_DIR}
    return _boot
