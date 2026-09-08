#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_PREFIX="${LIVETALKING_ENV_PREFIX:-$PROJECT_ROOT/../envs/livetalking}"
AVATARFORCING_DIR="$PROJECT_ROOT/AvatarForcing"
AVATARFORCING_COMMIT="507e3046ad539c938e0412ac7a74cf6e3ba2a61c"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda/Miniforge first." >&2
  exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.10 ffmpeg
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install \
  torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124
"$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements-avatarforcing.txt"

if [[ ! -e "$AVATARFORCING_DIR/.git" ]]; then
  if [[ -e "$AVATARFORCING_DIR" ]]; then
    echo "AvatarForcing exists but is not a Git checkout: $AVATARFORCING_DIR" >&2
    exit 1
  fi
  git clone https://github.com/TaekyungKi/AvatarForcing.git "$AVATARFORCING_DIR"
fi

current_commit="$(git -C "$AVATARFORCING_DIR" rev-parse HEAD)"
if [[ "$current_commit" != "$AVATARFORCING_COMMIT" ]]; then
  if [[ -n "$(git -C "$AVATARFORCING_DIR" status --porcelain)" ]]; then
    echo "AvatarForcing has local changes; refusing to switch commits." >&2
    exit 1
  fi
  git -C "$AVATARFORCING_DIR" fetch origin "$AVATARFORCING_COMMIT"
  git -C "$AVATARFORCING_DIR" checkout --detach "$AVATARFORCING_COMMIT"
fi

apply_patch_once() {
  local patch_file="$1"
  if git -C "$AVATARFORCING_DIR" apply --check "$patch_file" 2>/dev/null; then
    git -C "$AVATARFORCING_DIR" apply "$patch_file"
  elif git -C "$AVATARFORCING_DIR" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "Already applied: $patch_file"
  else
    echo "Patch does not apply cleanly: $patch_file" >&2
    exit 1
  fi
}

apply_patch_once "$PROJECT_ROOT/release/avatarforcing/patches/avatarforcing-runtime.patch"
apply_patch_once "$PROJECT_ROOT/release/avatarforcing/patches/avatarforcing-talking-only.patch"

echo "Environment ready: $ENV_PREFIX"
echo "Next: PYTHON_BIN=$PYTHON_BIN bash scripts/avatarforcing/download_weights.sh"
