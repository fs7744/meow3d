"""把主项目 `2d-game-meow` 的 Qwen-Image-2.1 权重以**硬链接**接到 meow3d。

**主项目目录严格只读**：本脚本只调用 os.link(src, dst)，不打开源文件写入、
不删除、不改名、不创建任何文件到主项目。硬链接只增加源 inode 的链接计数，
文件内容与目录条目完全不变。同卷（D:）不占额外磁盘空间。

    python tools/link_qwen.py                 # 用任一 python 都行
    python tools/link_qwen.py --verify       # 只校验，不建链接
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

SRC_ROOT = Path(r"D:\code\2d-game-meow\models")
DST_ROOT = Path(r"D:\code\meow3d\models\qwen")

# (源相对路径, 目标相对路径)；目录会递归整棵硬链接
TREES = [
    ("checkpoints/qwen-image-2.1", "checkpoints/qwen-image-2.1"),
    ("controlnet", "controlnet"),
    ("lora", "lora"),
]
SKIP_PARTS = {".cache", "__pycache__"}
SKIP_SUFFIX = {".lock", ".metadata", ".pyc"}

# controlnet / lora 里不需要的（NSFW 之类、超大且用不上）
SKIP_NAMES = {
    "NSFW Qwen by TheseAlpacas V2.safetensors",
    "qwen21-nsfw-image-edit.safetensors",
    "qwen2_1-femaledVaginus-nsfw-v1_3-fine_L22-31.safetensors",
    "nsfw-posture-v3.safetensors",
}


def _skip(p: Path) -> bool:
    return (any(part in SKIP_PARTS for part in p.parts)
            or p.suffix.lower() in SKIP_SUFFIX
            or p.name in SKIP_NAMES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if not SRC_ROOT.is_dir():
        print("找不到主项目 models:", SRC_ROOT)
        return 2

    total = linked = copied = skipped = bad = 0
    for src_rel, dst_rel in TREES:
        src_dir, dst_dir = SRC_ROOT / src_rel, DST_ROOT / dst_rel
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.rglob("*")):
            if not src.is_file() or _skip(src):
                continue
            rel = src.relative_to(src_dir)
            dst = dst_dir / rel
            total += 1
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            if args.verify:
                print("  缺:", dst)
                bad += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()          # 只删 meow3d 这边的旧链接
            try:
                os.link(src, dst)     # 源：只读，不写入
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1

    print(f"源文件 {total} 个；新建硬链接 {linked}，复制 {copied}，已存在跳过 {skipped}")
    if args.verify:
        print("缺", bad, "个")
        return 1 if bad else 0

    # 自检：关键文件都在，且确实是硬链接（同 inode）
    must = [
        "checkpoints/qwen-image-2.1/model_index.json",
        "checkpoints/qwen-image-2.1/transformer-fp8/model.safetensors",
        "checkpoints/qwen-image-2.1/text_encoder-fp8/model.safetensors",
        "checkpoints/qwen-image-2.1/vae/diffusion_pytorch_model.safetensors",
        "lora/qwen21_lightning.safetensors",
        "lora/qwen21_anyangle.safetensors",
        "controlnet/qwen21_union.safetensors",
    ]
    ok = True
    for rel in must:
        d = DST_ROOT / rel
        s = SRC_ROOT / rel
        exists = d.is_file()
        same = exists and d.stat().st_size == s.stat().st_size
        print(f"  [{'OK' if same else '!!'}] {rel}")
        ok &= bool(same)
    print("主项目目录未被写入（只创建了指向源 inode 的新目录项）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
