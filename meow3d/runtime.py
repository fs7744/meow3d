"""把 ComfyUI 的运行时和 ComfyUI-Trellis2 节点包挂进**当前进程**。

不启动 ComfyUI 服务、不加载它的全部内置节点（`import nodes` 会拖进整套
SD/采样器/模型补丁链）。这里只提供 TRELLIS.2 节点真正 import 的那几个模块：
`folder_paths` / `node_helpers` / `comfy.model_management` / `comfy.utils` /
`comfy_api.latest.Types`。

注意顺序：`folder_paths` 在 import 时就把 base_path / models_dir 算死了，
所以路径覆盖必须发生在它之前（改 `comfy.cli_args.args`）。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
COMFY_DIR = VENDOR / "ComfyUI"
TRELLIS_DIR = VENDOR / "ComfyUI-Trellis2"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "out"
INPUT_DIR = ROOT / "in"
TMP_DIR = ROOT / ".tmp"

_boot: Optional[Dict[str, Any]] = None


class Meow3DError(RuntimeError):
    """环境或生成链路不可用。消息面向用户，别塞堆栈。"""


def available() -> Dict[str, Any]:
    """只做静态检查（不加载模型、不 import torch），返回 {ok, reasons, ...}。"""
    reasons = []
    if not COMFY_DIR.is_dir():
        reasons.append(f"缺 ComfyUI 运行时：{COMFY_DIR}")
    if not TRELLIS_DIR.is_dir():
        reasons.append(f"缺 ComfyUI-Trellis2 节点包：{TRELLIS_DIR}")
    # 节点只按 models/<repo> 找权重，名字不能改
    trellis_ckpts = MODELS_DIR / "microsoft" / "TRELLIS.2-4B" / "ckpts"
    if not (trellis_ckpts / "ss_flow_img_dit_1_3B_64_bf16.safetensors").is_file():
        reasons.append(f"缺 TRELLIS.2-4B 权重：{trellis_ckpts}")
    dino = MODELS_DIR / "facebook" / "dinov3-vitl16-pretrain-lvd1689m" / "model.safetensors"
    if not dino.is_file():
        reasons.append(f"缺 DINOv3 权重：{dino}")
    return {"ok": not reasons, "reasons": reasons,
            "comfy": str(COMFY_DIR), "trellis": str(TRELLIS_DIR),
            "models": str(MODELS_DIR)}


def _install_server_stub() -> None:
    """有些 comfy 模块在 import 阶段就摸 PromptServer.instance。"""
    if "server" in sys.modules:
        return
    server = ModuleType("server")

    class _StubServer:
        instance = None

        def send_sync(self, *args: Any, **kwargs: Any) -> None:
            pass

        def send(self, *args: Any, **kwargs: Any) -> None:
            pass

    _StubServer.instance = _StubServer()
    server.PromptServer = _StubServer
    sys.modules["server"] = server


def _install_triton_stub_if_needed() -> None:
    """Windows 上可能没装 triton；节点只在换缓存目录时用它，不参与计算。"""
    try:
        import triton  # noqa: F401
        import triton.compiler  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass
    triton = ModuleType("triton")
    compiler = ModuleType("triton.compiler")
    triton.compiler = compiler
    sys.modules["triton"] = triton
    sys.modules["triton.compiler"] = compiler


def _install_flash_attn_shim() -> None:
    """Windows 上没有 flash-attn 轮子；注入一个 API 兼容的 SDPA 实现。

    visualbruno 的 `sparse/attention/full_attn.py` 有 SDPA 兜底，但
    `windowed_attn.py` **没有**：它无条件 `import flash_attn` 并直接调用
    varlen 接口，且 ATTN 不是 xformers/flash_attn 时 `attn_func_args` 根本
    不会被赋值。所以保留 `sparse_backend='flash_attn'`，用本 shim 顶上。

    数学上就是 scaled_dot_product_attention，按 cu_seqlens 分段算，
    与 flash-attn 等价；代价是 Python 循环（窗口多时会慢一些）。
    真装了 flash-attn 时这里直接放行，用原生实现。
    """
    try:
        import flash_attn  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass

    import torch
    import torch.nn.functional as F

    from importlib.machinery import ModuleSpec

    m = ModuleType("flash_attn")
    # transformers 用 importlib.util.find_spec("flash_attn") 探测：
    # sys.modules 里的模块若 __spec__ is None 会直接抛 ValueError。
    # 版本故意报 0.0.0，让 transformers 判定「有包但太旧」→ 不启用 FA2 路径，
    # 走它自己的 SDPA；节点侧则照常用本 shim 的 varlen 接口。
    m.__spec__ = ModuleSpec("flash_attn", loader=None)
    m.__version__ = "0.0.0"
    _cu_cache: Dict[int, List[int]] = {}

    def _cu_list(cu: Any) -> List[int]:
        if not torch.is_tensor(cu):
            return [int(x) for x in cu]
        key = (cu.data_ptr(), int(cu.numel()))
        if key not in _cu_cache:
            _cu_cache[key] = [int(x) for x in cu.detach().cpu().tolist()]
        return _cu_cache[key]

    def _varlen(q, k, v, cu_q, cu_k):
        outs = []
        cq, ck = _cu_list(cu_q), _cu_list(cu_k)
        for i in range(len(cq) - 1):
            a, b = cq[i], cq[i + 1]
            c, d = ck[i], ck[i + 1]
            if b <= a or d <= c:
                continue
            qi = q[a:b].transpose(0, 1).unsqueeze(0)      # [1,H,L,C]
            ki = k[c:d].transpose(0, 1).unsqueeze(0)
            vi = v[c:d].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qi, ki, vi)
            outs.append(o.squeeze(0).transpose(0, 1))      # [L,H,C]
        if not outs:
            return q.new_zeros((0,) + q.shape[1:])
        return torch.cat(outs, dim=0)

    def _pick_cu(pos, kw, names):
        if pos is not None:
            return pos
        for n in names:
            if n in kw:
                return kw[n]
        return None

    def varlen_qkvpacked_func(qkv, cu_seqlens_q=None, *a, **kw):
        q, k, v = qkv.unbind(dim=1)
        cu_q = _pick_cu(cu_seqlens_q, kw, ("cu_seqlens_q", "cu_seqlens"))
        cu_k = _pick_cu(kw.get("cu_seqlens_k"), kw, ("cu_seqlens_k",))
        if cu_k is None:
            cu_k = cu_q
        return _varlen(q, k, v, cu_q, cu_k)

    def varlen_kvpacked_func(q, kv, cu_seqlens_q=None, cu_seqlens_k=None, *a, **kw):
        k, v = kv.unbind(dim=1)
        cu_q = _pick_cu(cu_seqlens_q, kw, ("cu_seqlens_q", "cu_seqlens"))
        cu_k = _pick_cu(cu_seqlens_k, kw, ("cu_seqlens_k",))
        if cu_k is None:
            cu_k = cu_q
        return _varlen(q, k, v, cu_q, cu_k)

    def varlen_func(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, *a, **kw):
        if cu_seqlens_k is None:
            cu_seqlens_k = cu_seqlens_q
        return _varlen(q, k, v, cu_seqlens_q, cu_seqlens_k)

    def _dense(q, k, v, *a, **kw):
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2)

    def qkvpacked_func(qkv, *a, **kw):
        q, k, v = qkv.unbind(dim=2)
        return _dense(q, k, v)

    def kvpacked_func(q, kv, *a, **kw):
        k, v = kv.unbind(dim=2)
        return _dense(q, k, v)

    m.flash_attn_varlen_qkvpacked_func = varlen_qkvpacked_func
    m.flash_attn_varlen_kvpacked_func = varlen_kvpacked_func
    m.flash_attn_varlen_func = varlen_func
    m.flash_attn_qkvpacked_func = qkvpacked_func
    m.flash_attn_kvpacked_func = kvpacked_func
    m.flash_attn_func = _dense
    sys.modules["flash_attn"] = m


def _install_comfy_light_stubs() -> None:
    """给 comfy.model_management / comfy.utils 装轻量替身。

    新版 ComfyUI 的这两个模块会拖进 comfy_aimdo、comfy_kitchen、transformers、
    comfy.model_patcher 等一整条链，而 TRELLIS.2 节点实际只用到：
    mm.current_loaded_models / get_torch_device / soft_empty_cache /
    free_memory / get_total_memory，以及 utils.ProgressBar /
    load_torch_file / common_upscale。这里给出等价实现，让项目不必安装
    ComfyUI 的全套 SD 依赖。
    """
    import torch
    import comfy  # 已被 cli_args 导入过，拿到包对象

    mm = ModuleType("comfy.model_management")
    mm.current_loaded_models = []

    def _device():
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

    mm.get_torch_device = _device
    mm.get_total_memory = lambda *a, **k: (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)
    mm.free_memory = lambda *a, **k: (
        torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0)
    mm.soft_empty_cache = lambda *a, **k: (
        torch.cuda.empty_cache() if torch.cuda.is_available() else None)
    mm.unload_all_models = lambda *a, **k: None
    mm.intermediate_device = lambda *a, **k: torch.device("cpu")
    sys.modules["comfy.model_management"] = mm
    setattr(comfy, "model_management", mm)

    utils = ModuleType("comfy.utils")

    class ProgressBar:
        """宽容的进度条：节点/pipeline 只调 update*，不做展示。"""
        def __init__(self, total=None, *a, **k):
            self.total = total
            self.current = 0

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
        # ComfyUI 的 samples 是 [B,H,W,C]，F.interpolate 要 [B,C,H,W]
        x = samples.movedim(-1, 1)
        mode = {"nearest": "nearest", "bilinear": "bilinear",
                "bicubic": "bicubic", "area": "area"}.get(upscale_method, "bicubic")
        align = None if mode in ("nearest", "area") else False
        y = torch.nn.functional.interpolate(x, size=(height, width), mode=mode,
                                            align_corners=align)
        return y.movedim(1, -1)

    utils.ProgressBar = ProgressBar
    utils.load_torch_file = load_torch_file
    utils.common_upscale = common_upscale
    sys.modules["comfy.utils"] = utils
    setattr(comfy, "utils", utils)


def _install_comfy_api_stub() -> None:
    """导出节点只要 `Types.File3D` 包装一下路径。

    真的 `comfy_api.latest` 会拉 comfy_execution / PyAV / 视频编解码等一整条链；
    这里只给这个数据类，其余保持 sys.modules 原样。
    """
    pkg = ModuleType("comfy_api")
    latest = ModuleType("comfy_api.latest")

    class File3D:
        def __init__(self, path: str):
            self.path = path

        def __str__(self) -> str:
            return self.path

    # 类体不构成闭包，直接引用函数局部的 File3D 会 NameError
    Types = type("Types", (), {"File3D": File3D})
    latest.Types = Types
    pkg.latest = latest
    sys.modules.setdefault("comfy_api", pkg)
    sys.modules["comfy_api.latest"] = latest


def bootstrap() -> Dict[str, Any]:
    """幂等。→ {nodes, mappings, folder_paths, ...}"""
    global _boot
    if _boot is not None:
        return _boot

    st = available()
    if not st["ok"]:
        raise Meow3DError("；".join(st["reasons"]))

    for d in (OUT_DIR, INPUT_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. sys.path：ComfyUI 在前，让 folder_paths / comfy.* 可 import
    for p in (str(COMFY_DIR), str(VENDOR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 2. 关键：路径覆盖必须在 import folder_paths 之前
    import comfy.cli_args as cli_args
    cli_args.args.base_directory = str(ROOT)
    cli_args.args.models_directory = str(MODELS_DIR)
    cli_args.args.output_directory = str(OUT_DIR)
    cli_args.args.input_directory = str(INPUT_DIR)
    cli_args.args.temp_directory = str(TMP_DIR)

    _install_server_stub()
    _install_triton_stub_if_needed()
    _install_flash_attn_shim()
    _install_comfy_light_stubs()
    _install_comfy_api_stub()

    import folder_paths
    # 再兜一层：有些版本在别处重算
    folder_paths.output_directory = str(OUT_DIR)
    folder_paths.input_directory = str(INPUT_DIR)
    folder_paths.temp_directory = str(TMP_DIR)
    folder_paths.base_path = str(ROOT)
    folder_paths.models_dir = str(MODELS_DIR)

    # 3. 目录名带连字符，不能 import；用 importlib 以合法包名装载，
    #    这样 nodes.py 里的 `from .trellis2...` 相对导入才有包上下文。
    pkg_name = "comfyui_trellis2"
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            TRELLIS_DIR / "__init__.py",
            submodule_search_locations=[str(TRELLIS_DIR)],
        )
        if spec is None or spec.loader is None:
            raise Meow3DError(f"无法装载节点包：{TRELLIS_DIR}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = mod
        spec.loader.exec_module(mod)

    nodes = sys.modules[f"{pkg_name}.nodes"]
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})

    required = (
        "Trellis2LoadModel", "Trellis2PreProcessImage", "Trellis2ImageCondGenerator",
        "Trellis2SparseGenerator", "Trellis2ShapeGenerator",
        "Trellis2ShapeCascadeGenerator", "Trellis2DecodeLatents",
        "Trellis2FillHolesWithCuMesh", "Trellis2ReconstructMeshWithQuad",
        "Trellis2SimplifyMesh", "Trellis2FillHolesNicelyWithMeshlib",
        "Trellis2MeshWithVoxelToTrimesh", "Trellis2MeshTexturing",
        "Trellis2ExportMesh",
    )
    missing = [n for n in required if n not in mappings]
    if missing:
        raise Meow3DError("节点未注册：" + "、".join(missing))

    _boot = {
        "nodes": nodes,
        "mappings": mappings,
        "folder_paths": folder_paths,
        "output": OUT_DIR,
        "input": INPUT_DIR,
        "models": MODELS_DIR,
    }
    return _boot
