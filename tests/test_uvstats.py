"""UV 覆盖统计的单测：重点是别再把「岛外虚高」当成画质提升。"""
from __future__ import annotations

import struct

import numpy as np

from meow3d.projection import glbpatch, uvstats


def _tiny_uv_glb(path) -> None:
    """最小带 UV 的 GLB；indices 故意放在 primitive 顶层（Blender 的写法）。"""
    pos = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    uv = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
    idx = struct.pack("<3I", 0, 1, 2)

    def pad4(b):
        return b + b"\x00" * ((-len(b)) % 4)

    parts = [pad4(pos), pad4(uv), pad4(idx)]
    offsets, off = [], 0
    for p in parts:
        offsets.append(off)
        off += len(p)
    bin_ = b"".join(parts)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(uv)},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(idx)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": 3, "type": "SCALAR"},
        ],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
            "indices": 2,
        }]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    glbpatch._write_glb(path, gltf, bin_)


def test_read_glb_uv_handles_top_level_indices(tmp_path):
    p = tmp_path / "t.glb"
    _tiny_uv_glb(p)
    got = uvstats.read_glb_uv(p)
    assert got is not None, "indices 不在 attributes 里，旧实现会静默返回 None"
    uv, faces = got
    assert uv.shape == (3, 2)
    assert faces.shape == (1, 3)
    assert np.allclose(uv[1], [1.0, 0.0])


def test_read_glb_uv_missing_file_returns_none():
    assert uvstats.read_glb_uv("definitely_not_here.glb") is None


def test_rasterize_islands_covers_triangle():
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    m = uvstats.rasterize_islands(uv, np.array([[0, 1, 2]]), 64)
    assert 0.3 < m.mean() < 0.7        # 直角三角形约占半张图集
    assert not m[-1, -1]


def test_coverage_stats_separates_real_and_ghost(tmp_path):
    p = tmp_path / "t.glb"
    _tiny_uv_glb(p)
    isl = uvstats.island_mask(p, 64)
    assert isl is not None
    # 只涂岛外：全图有覆盖，但岛内为 0 —— 渲染时完全看不到
    cov = np.zeros((64, 64), bool)
    cov[~isl] = True
    st = uvstats.coverage_stats(cov, isl)
    assert 0.4 < st["coverage"] < 0.6
    assert st["coverage_uv"] == 0.0
    assert st["coverage_ghost"] == 1.0
    assert st["uv_usage"] == float(isl.mean())
    # 反过来：只涂岛内，虚高必须是 0
    st2 = uvstats.coverage_stats(isl, isl)
    assert st2["coverage_uv"] == 1.0
    assert st2["coverage_ghost"] == 0.0


def test_coverage_stats_without_islands():
    st = uvstats.coverage_stats(np.ones((8, 8), bool), None)
    assert st["coverage"] == 1.0
    assert st["uv_usage"] is None
    assert st["coverage_uv"] is None
