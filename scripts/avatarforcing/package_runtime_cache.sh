#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_DIR="$PROJECT_ROOT/cache/avatarforcing_tts"
OUTPUT_DIR="$PROJECT_ROOT/release/artifacts"
OUTPUT_FILE="$OUTPUT_DIR/avatarforcing_tts_cache_v1.tar.gz"

if [[ ! -d "$SOURCE_DIR/female" || ! -d "$SOURCE_DIR/male" ]]; then
  echo "Complete female/male AvatarForcing caches are required: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -czf "$OUTPUT_FILE" -C "$PROJECT_ROOT" cache/avatarforcing_tts
sha256sum "$OUTPUT_FILE"
echo "Upload this file as a repository Release asset: $OUTPUT_FILE"
