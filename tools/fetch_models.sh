#!/usr/bin/env bash
# 从 ModelScope（魔塔）拉取运行所需权重 —— 模型一律优先魔塔，不走 HuggingFace。
# 用法: bash tools/fetch_models.sh [trellis2native|trellis|dinov3|skintokens|all]
# 特性: 断点续传、已存在且大小一致则跳过、失败重试 3 次、自动挑代理。
#
# 代理（2026-11-05 实测，魔塔 SkinTokens 2MB range）：
#   直连                     2.25 MB/s
#   http://127.0.0.1:7897   3.64 MB/s   ← 最快，自动优先
#   http://127.0.0.1:52563  0.16 MB/s   （WorkBuddy 内置代理，备用）
#   hf-mirror 经 7897       0.35 MB/s   ← 比魔塔慢一个数量级，别用
# 可用 MS_PROXY=<url> 手动指定，MS_PROXY=none 强制直连。
set -u

PY="D:/code/meow3d/.venv/Scripts/python.exe"
MODELS="D:/code/meow3d/models"
MS="https://modelscope.cn/api/v1/models"

# curl 的代理参数（数组，可能为空）
PROXY_ARGS=()
pick_proxy() {
  if [ -n "${MS_PROXY:-}" ]; then
    if [ "$MS_PROXY" = "none" ]; then PROXY_ARGS=(); echo "  代理: 强制直连"
    else PROXY_ARGS=(-x "$MS_PROXY"); echo "  代理: $MS_PROXY (手动)"; fi
    return
  fi
  local p
  for p in http://127.0.0.1:7897 http://127.0.0.1:52563; do
    if curl -s --max-time 3 -x "$p" -o /dev/null "https://www.modelscope.cn" 2>/dev/null; then
      PROXY_ARGS=(-x "$p"); echo "  代理: $p"; return
    fi
  done
  PROXY_ARGS=(); echo "  代理: 无（直连）"
}

fetch_repo() {
  local repo="$1" dest="$2"
  echo "=== $repo -> $dest ==="
  mkdir -p "$dest"

  local list
  list=$(PYTHONIOENCODING=utf-8 "$PY" - "$MS" "$repo" <<'PYEOF'
import json, sys, urllib.request
base, repo = sys.argv[1], sys.argv[2]
url = f"{base}/{repo}/repo/files?Recursive=true"
d = json.load(urllib.request.urlopen(url, timeout=30))
for f in d.get("Data", {}).get("Files", []):
    if f.get("Type") != "blob":
        continue
    print(f"{f['Path']}\t{f.get('Size', 0) or 0}")
PYEOF
)
  if [ -z "$list" ]; then echo "  !! 拿不到文件清单"; return 1; fi

  local total=0 ok_n=0 skip_n=0 fail=0
  while IFS=$'\t' read -r path size; do
    # Python 在 Windows 的 stdout 是 CRLF，size 会带 \r；不去掉会让
    # "2178" != "2178\r"，明明下对了却报失败（2026-11-05 踩过）。
    size=${size%$'\r'}
    path=${path%$'\r'}
    [ -z "$path" ] && continue
    total=$((total+1))
    local out="$dest/$path"
    local url="https://modelscope.cn/models/$repo/resolve/master/$path"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ]; then
      local cur
      cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
      if [ "$cur" = "$size" ]; then skip_n=$((skip_n+1)); continue; fi
    fi
    echo "  -> $path ($size bytes)"

    # 已存在但不完整才续传；魔塔部分小文件不支持 Range，续传失败就删掉重下。
    local range=()
    [ -f "$out" ] && range=(-C -)
    curl -L "${range[@]}" --retry 3 --retry-delay 3 --max-time 3600 \
         "${PROXY_ARGS[@]}" -o "$out" "$url" \
         -w "     %{http_code} %{size_download}B %{speed_download}B/s %{time_total}s\n"
    local rc=$?
    local fin
    fin=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$rc" != "0" ] || [ "$fin" != "$size" ]; then
      echo "     .. 续传/首下不完整，删除后完整重下"
      rm -f "$out"
      curl -L --retry 3 --retry-delay 3 --max-time 3600 \
           "${PROXY_ARGS[@]}" -o "$out" "$url" \
           -w "     %{http_code} %{size_download}B %{speed_download}B/s %{time_total}s\n"
      rc=$?
      fin=$(stat -c%s "$out" 2>/dev/null || echo 0)
    fi
    if [ "$rc" != "0" ] || [ "$fin" != "$size" ]; then
      echo "     !! 失败 rc=$rc 期望=$size 实际=$fin"
      fail=$((fail+1))
    else
      ok_n=$((ok_n+1))
    fi
  done <<< "$list"
  echo "=== $repo 完成: 新下 $ok_n / 跳过 $skip_n / 失败 $fail (共 $total) ==="
  return $fail
}

# Trellis2LoadModel 会**无条件**检查这两个旧版 TRELLIS-image-large 文件是否存在，
# 缺了就直连 huggingface.co 下载（国内代理容易 502）。只下这两个，不要整个 repo。
fetch_image_large() {
  local repo="microsoft/TRELLIS-image-large"
  local dest="$MODELS/microsoft/TRELLIS-image-large/ckpts"
  mkdir -p "$dest"
  echo "=== $repo (仅 2 个 ckpts 文件) ==="
  for f in ss_dec_conv3d_16l8_fp16.json ss_dec_conv3d_16l8_fp16.safetensors; do
    local out="$dest/$f"
    if [ -s "$out" ]; then echo "  $f 已存在，跳过"; continue; fi
    curl -sL -C - --retry 3 --max-time 900 "${PROXY_ARGS[@]}" -o "$out" \
         "https://modelscope.cn/models/$repo/resolve/master/ckpts/$f" \
         -w "  $f  %{http_code} %{size_download}B %{speed_download}B/s\n"
  done
}

# SkinTokens 自动绑定权重（共约 1.6 GB）。
# 直接落到上游源码的 experiments/ 下：官方 demo.py 的
# MODEL_CKPTS = ["experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"]
# 是相对 cwd=repo root 的路径，放这里才能原样跑。vendor/ 已被 gitignore，
# 大权重不会进仓库。
fetch_skintokens() {
  fetch_repo "VAST-AI-Research/SkinTokens" \
             "D:/code/meow3d/vendor/SkinTokens"
}

# StableGen v0.5.0 的 TRELLIS.2 内置节点链要 5 个**单文件**权重（约 13.1GB）。
# 与旧的自定义节点链（microsoft/TRELLIS.2-4B 的 ckpts 分件）不是一套，
# 文件名和格式都不同。ComfyUI 按 folder_paths 在 models/<子目录> 下找。
fetch_trellis2_native() {
  local items=(
    "Comfy-Org/TRELLIS.2|diffusion_models/trellis_2_bf16.safetensors|10338"
    "Comfy-Org/TRELLIS.2|vae/trellis_2_shape_vae_bf16.safetensors|1096"
    "Comfy-Org/TRELLIS.2|vae/trellis_2_texture_vae_bf16.safetensors|948"
    "Comfy-Org/TRELLIS.2|clip_vision/dino_v3_vit_l.safetensors|1213"
    "Comfy-Org/BiRefNet|background_removal/birefnet.safetensors|445"
  )
  local it repo path mb out want cur
  for it in "${items[@]}"; do
    repo="${it%%|*}"; it="${it#*|}"
    path="${it%%|*}"; mb="${it##*|}"
    out="$MODELS/$path"; want=$((mb * 1048576))
    mkdir -p "$(dirname "$out")"
    cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
    # 魔塔给的是十进制 MB，留 2% 余量判断完成
    if [ "$cur" -ge $((want * 98 / 100)) ] && [ "$cur" -le $((want * 102 / 100)) ]; then
      echo "  $path 已存在（$((cur / 1048576)) MB），跳过"; continue
    fi
    echo "  -> $path (~${mb} MB)"
    local range=()
    [ -f "$out" ] && [ "$cur" -gt 0 ] && range=(-C -)
    curl -L "${range[@]}" --retry 3 --retry-delay 3 --max-time 7200 \
         "${PROXY_ARGS[@]}" -o "$out" \
         "https://modelscope.cn/models/$repo/resolve/master/$path" \
         -w "     %{http_code} %{size_download}B %{speed_download}B/s %{time_total}s\n"
  done
  echo "=== 结果 ==="
  find "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/clip_vision" \
       "$MODELS/background_removal" -type f -printf "%p  %.0f MB\n" 2>/dev/null
}

pick_proxy
case "${1:-all}" in
  trellis2native) fetch_trellis2_native ;;
  trellis)
    fetch_repo "microsoft/TRELLIS.2-4B" "$MODELS/microsoft/TRELLIS.2-4B"
    fetch_image_large
    ;;
  dinov3)  fetch_repo "facebook/dinov3-vitl16-pretrain-lvd1689m" "$MODELS/facebook/dinov3-vitl16-pretrain-lvd1689m" ;;
  skintokens) fetch_skintokens ;;
  all)
    fetch_repo "microsoft/TRELLIS.2-4B" "$MODELS/microsoft/TRELLIS.2-4B"
    fetch_repo "facebook/dinov3-vitl16-pretrain-lvd1689m" "$MODELS/facebook/dinov3-vitl16-pretrain-lvd1689m"
    fetch_image_large
    ;;
  *) echo "用法: $0 [trellis2native|trellis|dinov3|skintokens|all]"; exit 2 ;;
esac
