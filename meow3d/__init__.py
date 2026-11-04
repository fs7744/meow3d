"""meow3d —— 独立的「图生 3D（带原生纹理）」项目。

改的是 visualbruno/ComfyUI-Trellis2 的 TRELLIS.2 原生纹理链，
在**项目进程内**执行 ComfyUI 节点图，不启动 ComfyUI 服务。
需要 Python 3.12 + torch 2.8.0+cu128（cp312 编译轮子）。
"""
from .api import generate
from .runtime import Meow3DError, ROOT, available, bootstrap

__all__ = ["ROOT", "Meow3DError", "available", "bootstrap", "generate"]
