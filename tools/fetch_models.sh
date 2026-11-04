#!/usr/bin/env bash
# 从 ModelScope 拉取 TRELLIS.2 运行所需权重。
# 用法: bash tools/fetch_models.sh [trellis|dinov3|all]
# 特性: 断点续传、已存在且大小一致则跳过、失败重试 3 次。
set -u

PY="D:/code/meow3d/.venv/Scripts/python.exe"
MODELS="D:/code/meow3d/models"
MS="https://modelscope.cn/api/v1/models"

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
    [ -z "$path" ] && continue
    total=$((total+1))
    local out="$dest/$path"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ]; then
      local cur
      cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
      if [ "$cur" = "$size" ]; then skip_n=$((skip_n+1)); continue; fi
    fi
    echo "  -> $path ($size bytes)"
    curl -L -C - --retry 3 --retry-delay 3 --max-time 3600 \
         -o "$out" "https://modelscope.cn/models/$repo/resolve/master/$path" \
         -w "     %{http_code} %{size_download}B %{speed_download}B/s %{time_total}s\n"
    local rc=$? fin
    fin=$(stat -c%s "$out" 2>/dev/null || echo 0)
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
    curl -sL -C - --retry 3 --max-time 900 -o "$out" \
         "https://modelscope.cn/models/$repo/resolve/master/ckpts/$f" \
         -w "  $f  %{http_code} %{size_download}B %{speed_download}B/s\n"
  done
}

case "${1:-all}" in
  trellis)
    fetch_repo "microsoft/TRELLIS.2-4B" "$MODELS/microsoft/TRELLIS.2-4B"
    fetch_image_large
    ;;
  dinov3)  fetch_repo "facebook/dinov3-vitl16-pretrain-lvd1689m" "$MODELS/facebook/dinov3-vitl16-pretrain-lvd1689m" ;;
  all)
    fetch_repo "microsoft/TRELLIS.2-4B" "$MODELS/microsoft/TRELLIS.2-4B"
    fetch_repo "facebook/dinov3-vitl16-pretrain-lvd1689m" "$MODELS/facebook/dinov3-vitl16-pretrain-lvd1689m"
    fetch_image_large
    ;;
  *) echo "用法: $0 [trellis|dinov3|all]"; exit 2 ;;
esac
