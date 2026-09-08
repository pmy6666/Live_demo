#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_PREFIX="${LIVETALKING_ENV_PREFIX:-$PROJECT_ROOT/../envs/livetalking}"
PYTHON_BIN="$ENV_PREFIX/bin/python"

echo "[1/5] Creating the pinned Python environment..."
LIVETALKING_ENV_PREFIX="$ENV_PREFIX" bash "$SCRIPT_DIR/setup_env.sh"

echo "[2/5] Downloading and verifying model weights..."
LIVETALKING_ENV_PREFIX="$ENV_PREFIX" PYTHON_BIN="$PYTHON_BIN" \
  bash "$SCRIPT_DIR/download_weights.sh"

echo "[3/5] Verifying the bundled offline speech cache..."
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" \
  scripts/avatarforcing/build_avatarforcing_tts_cache.py verify

echo "[4/5] Running the file-only release validation..."
"$PYTHON_BIN" start_avatarforcing.py --validate-only --files-only

echo "[5/5] Reporting the PyTorch and CUDA status..."
"$PYTHON_BIN" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU validation was not run. Check the NVIDIA driver before starting the demo.")
PY

echo
echo "Preparation completed."
echo "Validate GPU: $PYTHON_BIN $PROJECT_ROOT/start_avatarforcing.py --validate-only"
echo "Start demo:   $PYTHON_BIN $PROJECT_ROOT/start_avatarforcing.py"
