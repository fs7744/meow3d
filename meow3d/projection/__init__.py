"""projection-based texturing —— 不依赖 nvdiffrast / nvdiffrec 的贴图管线。

两条后端：
- `pipeline.run`  v1：UV texel 反向映射（保留 `replace` 模式，正面原图直贴）。
                     完整多相机混合在碎片 UV 岛上会碎，勿用于 blend。
- `bake.run`      v2：Blender 3D 表面投影 + bake（推荐）。投影在连续表面上完成，
                     UV 只是产物，因此不会碎；CLI 见 tools/bake_projection.py。
"""

from .pipeline import run
from . import bake

__all__ = ["run", "bake"]
