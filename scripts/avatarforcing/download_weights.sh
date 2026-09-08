#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_PREFIX="${LIVETALKING_ENV_PREFIX:-$PROJECT_ROOT/../envs/livetalking}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
PRETRAINED_DIR="$PROJECT_ROOT/AvatarForcing/pretrained_dir"
WAV2VEC_DIR="$PRETRAINED_DIR/wav2vec2-base-960h"
TORCH_CHECKPOINT_DIR="$PROJECT_ROOT/.runtime/torch/hub/checkpoints"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: $PYTHON_BIN" >&2
  echo "Run scripts/avatarforcing/setup_env.sh first." >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT/AvatarForcing/.git" ]]; then
  echo "AvatarForcing source missing. Run scripts/avatarforcing/setup_env.sh first." >&2
  exit 1
fi

mkdir -p "$PRETRAINED_DIR" "$WAV2VEC_DIR" "$TORCH_CHECKPOINT_DIR"

if [[ ! -f "$PRETRAINED_DIR/motion_autoencoder.pth" || ! -f "$PRETRAINED_DIR/flow_transformer.pth" ]]; then
  echo "Downloading official AvatarForcing checkpoints from Google Drive..."
  "$PYTHON_BIN" -m gdown --folder \
    "https://drive.google.com/drive/folders/1rN52J2QXD8A-r2CZ8nDqFdwvZuPRmMsc" \
    -O "$PRETRAINED_DIR/"
fi

wav2vec_missing=0
for required in config.json preprocessor_config.json vocab.json model.safetensors; do
  [[ -f "$WAV2VEC_DIR/$required" ]] || wav2vec_missing=1
done
if [[ "$wav2vec_missing" -eq 1 ]]; then
  echo "Downloading the minimal facebook/wav2vec2-base-960h snapshot..."
  "$PYTHON_BIN" - "$WAV2VEC_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/wav2vec2-base-960h",
    revision="22aad52d435eb6dbaf354bdad9b0da84ce7d6156",
    local_dir=sys.argv[1],
    allow_patterns=[
        "config.json",
        "feature_extractor_config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)
PY
else
  echo "Wav2Vec2 files already exist; verifying them."
fi

echo "Downloading face-alignment detector weights into project-local TORCH_HOME..."
"$PYTHON_BIN" - "$TORCH_CHECKPOINT_DIR" <<'PY'
import hashlib
import pathlib
import sys
import urllib.request

root = pathlib.Path(sys.argv[1])
artifacts = {
    "s3fd-619a316812.pth": (
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543",
    ),
    "2DFAN4-11f355bf06.pth.tar": (
        "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar",
        "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa",
    ),
}
root.mkdir(parents=True, exist_ok=True)
for name, (url, expected) in artifacts.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        urllib.request.urlretrieve(url, path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"checksum mismatch: {path}: {actual}")
    print(f"verified {name}: {actual}")
PY

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path")"
  actual="${actual%% *}"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch: $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 1
  fi
  echo "verified $(basename "$path"): $actual"
}

check_sha256 \
  "41391516d761a3768bf43ad4ce10f41c66b3a652394610c1be2bc9617f327c33" \
  "$PRETRAINED_DIR/motion_autoencoder.pth"
check_sha256 \
  "1c258fdcc534941ef2cf46a53188690305584ca642099c523db08122e05083cc" \
  "$PRETRAINED_DIR/flow_transformer.pth"
check_sha256 \
  "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a" \
  "$WAV2VEC_DIR/model.safetensors"

echo "All AvatarForcing weights are ready."
