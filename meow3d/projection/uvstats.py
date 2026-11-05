"""UV 图集覆盖统计：把「全图覆盖」和「UV 岛内有效覆盖」分开。

## 为什么需要这个

`bake()` 里的 `--margin`（Blender bake 边缘扩张）会把已投影颜色向外涂。
对 **碎片 UV**（TRELLIS.2 稠密网格经 smart unwrap 后有成千上万个微小岛），
这些扩张色绝大部分落在 **UV 岛外的空白 texel** —— 渲染时没有任何三角形
采样它们，"覆盖率"却从 26.6% 一路涨到 89.4%。

实测（Asuka 白模 / smart UV / 1024²）：

| margin | 全图覆盖 | UV 岛占用 | 岛内覆盖 |
|---|---|---|---|
| 4  | 26.6% | 14.7% | 79.0% |
| 16 | 57.2% | 14.7% | 83.9% |
| 32 | 78.8% | 14.7% | 88.2% |
| 64 | 89.4% | 14.7% | 89.8% |

四张 3D 预览**几乎逐像素相同**：真正有贡献的只有岛内那 79%→89.8%。
所以只看全图覆盖率会把「往空白处涂色」误判成画质提升。

本模块直接解析 GLB 的 `TEXCOORD_0` 与 indices，光栅化出 UV 岛 mask，
再和 `merge()` 写出的 `mask.png` 相交，给出三个可区分的数字。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

Array = np.ndarray

_DTYPE = {5120: "<i1", 5121: "<u1", 5122: "<i2",
          5123: "<u2", 5125: "<u4", 5126: "<f4"}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb_uv(path) -> Optional[tuple[Array, Array]]:
    """读 GLB 第一个 primitive 的 UV 与三角面；没有就返回 None。

    不用 trimesh：它对某些 Blender 导出的中间 GLB 会给出 `visual.uv=None`，
    而 GLB 里其实有 `TEXCOORD_0`。Draco 压缩的情况也直接返回 None（不致命）。
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = p.read_bytes()
        if len(data) < 20 or data[:4] != b"glTF":
            return None
        off = 12
        jlen, _ = struct.unpack_from("<II", data, off)
        off += 8
        gltf = json.loads(data[off:off + jlen])
        off += jlen
        blen, _ = struct.unpack_from("<II", data, off)
        off += 8
        blob = data[off:off + blen]

        if gltf.get("extensionsRequired") or any(
                "KHR_draco" in pr.get("extensions", {})
                for m in gltf.get("meshes", [])
                for pr in m.get("primitives", [])):
            return None
        prim = gltf["meshes"][0]["primitives"][0]
        attrs = prim.get("attributes", {})
        if "TEXCOORD_0" not in attrs or "indices" not in prim:
            return None

        def acc(i: int) -> Array:
            a = gltf["accessors"][i]
            bv = gltf["bufferViews"][a["bufferView"]]
            dt = _DTYPE.get(int(a["componentType"]))
            n = _NCOMP.get(a["type"])
            if dt is None or n is None:
                raise ValueError(f"不支持的 accessor {a}")
            start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            arr = np.frombuffer(blob, dtype=np.dtype(dt),
                                count=int(a["count"]) * n, offset=start)
            return arr.reshape(-1, n)

        uv = acc(attrs["TEXCOORD_0"])[:, :2].astype(np.float64)
        idx = acc(prim["indices"]).reshape(-1)
        faces = idx[:len(idx) // 3 * 3].reshape(-1, 3)
        return uv, faces
    except (KeyError, IndexError, ValueError, struct.error, json.JSONDecodeError):
        return None


def rasterize_islands(uv: Array, faces: Array, size: int) -> Array:
    """把 (N,2) UV 与 (M,3) 面索引光栅化成岛占用 mask。"""
    pts = np.clip((np.asarray(uv) * int(size)).astype(np.int32),
                  0, int(size) - 1)
    m = np.zeros((int(size), int(size)), np.uint8)
    # 一次性提交所有多边形，比逐面 fillConvexPoly 快一个数量级
    cv2.fillPoly(m, [pts[t] for t in np.asarray(faces)], 1)
    return m.astype(bool)


def island_mask(path, size: int) -> Optional[Array]:
    """读 GLB 并光栅化 UV 岛；读不到 UV 返回 None。"""
    got = read_glb_uv(path)
    if got is None:
        return None
    return rasterize_islands(got[0], got[1], size)


def coverage_stats(coverage: Array, islands: Optional[Array]) -> Dict[str, Any]:
    """返回三个口径的覆盖率。

    * ``coverage``      —— 全图被投影写到的 texel（旧口径，可能虚高）
    * ``uv_usage``      —— UV 岛本身占多少 texel（展开质量）
    * ``coverage_uv``   —— **岛内**被投影写到的比例（真正影响渲染）
    * ``coverage_ghost``—— 涂在岛外的比例（渲染时不被采样，纯浪费）
    """
    cov = np.asarray(coverage, bool)
    out: Dict[str, Any] = {
        "coverage": float(cov.mean()),
        "uv_usage": None, "coverage_uv": None, "coverage_ghost": None,
    }
    if islands is None:
        return out
    isl = np.asarray(islands, bool)
    if not isl.any():
        return out
    out["uv_usage"] = float(isl.mean())
    out["coverage_uv"] = float((cov & isl).sum() / isl.sum())
    outside = ~isl
    out["coverage_ghost"] = (float((cov & outside).sum() / outside.sum())
                             if outside.any() else 0.0)
    return out
