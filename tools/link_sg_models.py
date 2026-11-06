"""把现有 TRELLIS.2 权重按 StableGen 节点包的目录约定摆好（硬链接，0 复制）。

PozzettiAndrea/ComfyUI-TRELLIS2 pin 9b878516 的 `_init_config()`
(nodes/stages.py:176-244) 只认 `models/trellis2/`：

  pipeline.json
  ckpts/<model_name>.json + ckpts/<model_name>.safetensors

其中 `<model_name>` 来自 pipeline.json 的 `args.models`；另外硬编码还要
`ckpts/shape_enc_next_dc_f16c32_fp16`（stages.py:231-239）。

meow3d 早就有一模一样的权重（models/microsoft/TRELLIS.2-4B/ 与
TRELLIS-image-large/），文件名完全一致，只是目录不同。这里用硬链接复用，
15GB 不复制、不重下。DINOv3 同理映射到 models/dinov3/。

    python tools/link_sg_models.py            # 建链接
    python tools/link_sg_models.py --check    # 只核对，不写
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"
DST = MODELS / "trellis2"


def _link(src: Path, dst: Path, *, check: bool) -> str:
    if not src.is_file():
        return f"MISSING  {src}"
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return "ok"
    if check:
        return "need"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copyfile(src, dst)
        return "copied"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    trellis = MODELS / "microsoft" / "TRELLIS.2-4B"
    old = MODELS / "microsoft" / "TRELLIS-image-large"
    pipeline = trellis / "pipeline.json"
    if not pipeline.is_file():
        print("找不到", pipeline)
        return 2

    jobs: list[tuple[Path, Path]] = [(pipeline, DST / "pipeline.json")]
    names = set()
    cfg = json.loads(pipeline.read_text(encoding="utf-8"))
    for model_path in (cfg.get("args") or {}).get("models", {}).values():
        # "microsoft/TRELLIS-image-large/ckpts/xxx" → ckpts/xxx
        # "ckpts/xxx"                              → ckpts/xxx
        parts = model_path.split("/")
        name = "/".join(parts[2:]) if len(parts) >= 3 and not model_path.startswith("ckpts/") else model_path
        names.add(name)
    # stages.py:231 硬编码的 shape_slat_encoder
    names.add("ckpts/shape_enc_next_dc_f16c32_fp16")

    for name in sorted(names):
        stem = Path(name)
        for suffix in (".json", ".safetensors"):
            rel = stem.with_suffix(suffix)
            # 先在本仓库主模型找，再退到 TRELLIS-image-large
            src = trellis / rel
            if not src.is_file():
                src = old / rel
            jobs.append((src, DST / rel))

    # DINOv3：dinov3.py:322-339 用 models/dinov3/model.safetensors
    dino_src = MODELS / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    dino_dst = MODELS / "dinov3"
    if dino_src.is_dir():
        for f in dino_src.iterdir():
            if f.is_file():
                jobs.append((f, dino_dst / f.name))

    print(f"目标目录: {DST}")
    print(f"待处理 {len(jobs)} 个文件" + ("（只核对）" if args.check else ""))
    missing = 0
    changed = 0
    for src, dst in jobs:
        r = _link(src, dst, check=args.check)
        if r.startswith("MISSING"):
            missing += 1
            print("  !!", r)
        elif r in ("linked", "copied", "need"):
            changed += 1
            if changed <= 12:
                print(f"  {r:7} {dst.relative_to(MODELS)}")
    print(f"完成：已就位 {len(jobs) - missing - changed}，"
          f"新增 {changed}，缺失 {missing}")
    if missing:
        print("缺失的请先用 tools/fetch_models.sh 从魔塔补齐")
        return 1
    if args.check and changed:
        print("有未就位项，去掉 --check 再跑一次")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
