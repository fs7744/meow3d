"""二进制级 GLB 贴图替换：只动 baseColorTexture 的图片字节，网格一字节不改。

为什么不直接用 trimesh 重新导出：
  * trimesh 会把 GLB 拆开再拼回去，实测**丢掉 metallicRoughnessTexture**
    （原 GLB 有 2048² 的 MR 图，导出后变 None），金属度/粗糙度外观就变了。
  * Blender 导入导出更狠：按 loop 拆顶点，36 万顶点膨胀到 135 万，
    文件 21MB → 50MB，而 UV 没变（keep 模式）——纯浪费。

GLB 容器结构（glTF 2.0）：
    header  magic 'glTF' | version=2 | total length
    chunk0  length | 'JSON' | json（空格 0x20 补齐到 4 字节）
    chunk1  length | 'BIN\\0' | 二进制（0x00 补齐到 4 字节）

做法：定位 baseColorTexture → textures[i].source → images[j].bufferView，
把该 bufferView 的字节换掉，长度差是 4 的倍数（PNG 补齐），
因此后面所有 bufferView 的 byteOffset 平移后**仍然满足 4 字节对齐**。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


def _read_glb(path: str | Path) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", raw, 0)
    if magic != GLB_MAGIC:
        raise ValueError(f"{path} 不是 GLB（magic={magic:#x}）")
    if version != 2:
        raise ValueError(f"只支持 glTF 2.0 容器，读到 version={version}")
    if total != len(raw):
        raise ValueError(f"GLB 头声明 {total} 字节，实际 {len(raw)}")

    pos, gltf, binary = 12, None, b""
    while pos < len(raw):
        clen, ctype = struct.unpack_from("<II", raw, pos)
        data = raw[pos + 8: pos + 8 + clen]
        if ctype == CHUNK_JSON:
            gltf = json.loads(data.decode("utf-8"))
        elif ctype == CHUNK_BIN:
            binary = data
        pos += 8 + clen        # chunk 的 clen 已含补齐，不用再算
    if gltf is None:
        raise ValueError("GLB 缺 JSON chunk")
    return gltf, binary


def _write_glb(path: str | Path, gltf: dict, binary: bytes) -> None:
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    bn = binary + b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(js) + (8 + len(bn) if bn else 0)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, 2, total)
    out += struct.pack("<II", len(js), CHUNK_JSON) + js
    if bn:
        out += struct.pack("<II", len(bn), CHUNK_BIN) + bn
    Path(path).write_bytes(bytes(out))


def base_color_image_index(gltf: dict, material: Optional[int] = None) -> list[int]:
    """返回需要替换的 image 下标列表（去重）。

    material=None 表示所有带 baseColorTexture 的材质都换。
    """
    textures = gltf.get("textures", [])
    out: list[int] = []
    mats = gltf.get("materials", [])
    for mi, mat in enumerate(mats):
        if material is not None and mi != material:
            continue
        pbr = mat.get("pbrMetallicRoughness") or {}
        bct = pbr.get("baseColorTexture")
        if not bct:
            continue
        ti = bct.get("index")
        if ti is None or ti >= len(textures):
            continue
        src = textures[ti].get("source")
        if src is None or src >= len(gltf.get("images", [])):
            continue
        if src not in out:
            out.append(src)
    return out


def replace_image_bytes(gltf: dict, binary: bytearray, image_index: int,
                        png: bytes) -> int:
    """把 images[image_index] 的字节换成 png，返回长度增量。"""
    images = gltf.get("images", [])
    if image_index >= len(images):
        raise IndexError(f"image {image_index} 不存在")
    img = images[image_index]
    if "bufferView" not in img:
        raise ValueError("图片是外部 uri/data uri，不支持二进制替换")
    bvi = img["bufferView"]
    bvs = gltf["bufferViews"]
    if bvi >= len(bvs):
        raise IndexError(f"bufferView {bvi} 不存在")
    bv = bvs[bvi]
    offset = int(bv.get("byteOffset", 0))
    length = int(bv["byteLength"])
    end = offset + length

    # 替换区要连同**老图后面的对齐 padding** 一起吃掉：glTF 的 bufferView
    # 起点必须 4 字节对齐，所以图片后面通常有 0~3 个填充字节。只换 length 字节时
    # 长度增量不是 4 的倍数，后面所有 bufferView 的 offset 就会错位。
    later = [int(v.get("byteOffset", 0)) for v in bvs
             if v is not bv and int(v.get("byteOffset", 0)) >= end]
    region_end = min(min(later), len(binary)) if later else len(binary)
    region_len = region_end - offset
    # 让「新长度 % 4 == 老长度 % 4」→ delta 一定是 4 的倍数
    padded = png + b"\x00" * ((region_len - len(png)) % 4)
    delta = len(padded) - region_len
    binary[offset:region_end] = padded

    bv["byteLength"] = len(png)
    img["mimeType"] = "image/png"
    for other in bvs:
        if other is bv:
            continue
        o = int(other.get("byteOffset", 0))
        if o >= region_end:
            other["byteOffset"] = o + delta
    for buf in gltf.get("buffers", []):
        if "byteLength" in buf:
            buf["byteLength"] = len(binary)
    return delta


def patch_base_color(src: str | Path, png: str | Path | bytes,
                     dst: str | Path) -> dict:
    """把 src 的 baseColorTexture 换成 png，写出 dst；返回诊断信息。"""
    gltf, binary = _read_glb(src)
    data = png if isinstance(png, (bytes, bytearray)) else Path(png).read_bytes()
    idxs = base_color_image_index(gltf)
    if not idxs:
        raise ValueError("这个 GLB 没有 baseColorTexture，没法原地换图")
    buf = bytearray(binary)
    # 从后往前替换：避免前一次平移影响后面的 bufferView 定位
    total_delta = 0
    for i in sorted(idxs, reverse=True):
        total_delta += replace_image_bytes(gltf, buf, i, data)
    _write_glb(dst, gltf, bytes(buf))
    return {"images": idxs, "delta_bytes": total_delta,
            "src_size": Path(src).stat().st_size,
            "dst_size": Path(dst).stat().st_size}


def image_size(glb: str | Path, image_index: int = -1) -> Optional[tuple[int, int]]:
    """不用 PIL 解析贴图尺寸：直接读 PNG/JPEG 头，用于「图集边长自动等于原生尺寸」。"""
    gltf, binary = _read_glb(glb)
    idxs = base_color_image_index(gltf)
    if not idxs:
        return None
    if image_index < 0:
        image_index = idxs[0]
    bv = gltf["bufferViews"][gltf["images"][image_index]["bufferView"]]
    off = int(bv.get("byteOffset", 0))
    head = binary[off: off + 32]
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack_from(">II", head, 16)
        return int(w), int(h)
    # JPEG：扫 SOFn
    off2 = off
    end = off + int(bv["byteLength"])
    while off2 < end:
        if binary[off2] != 0xFF:
            break
        marker = binary[off2 + 1]
        seg = struct.unpack_from(">H", binary, off2 + 2)[0]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack_from(">HH", binary, off2 + 5)
            return int(w), int(h)
        off2 += 2 + seg
    return None
