#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_PREFIX="${LIVETALKING_ENV_PREFIX:-$PROJECT_ROOT/../envs/livetalking}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PREFIX/bin/python}"
CACHE_URL="${AVATARFORCING_CACHE_URL:-https://github.com/pmy6666/Live_demo/releases/download/avatarforcing-final-v1/avatarforcing_tts_cache_v1.tar.gz}"
EXPECTED_SHA256="${AVATARFORCING_CACHE_SHA256:-5e25ce6e854518933beadf310c625ff5d2b9659b4dfa38d254947165578eedeb}"
DOWNLOAD_DIR="$PROJECT_ROOT/.download_cache/avatarforcing"
ARCHIVE="$DOWNLOAD_DIR/avatarforcing_tts_cache_v1.tar.gz"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR"
"$PYTHON_BIN" - "$CACHE_URL" "$ARCHIVE" "$EXPECTED_SHA256" "$PROJECT_ROOT" <<'PY'
import hashlib
import pathlib
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request

url, archive_arg, expected, project_arg = sys.argv[1:]
archive = pathlib.Path(archive_arg)
project = pathlib.Path(project_arg).resolve()
parsed = urllib.parse.urlparse(url)
if parsed.scheme in {"", "file"}:
    source = pathlib.Path(urllib.request.url2pathname(parsed.path)).resolve()
    if source != archive.resolve():
        shutil.copyfile(source, archive)
else:
    urllib.request.urlretrieve(url, archive)

actual = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"cache checksum mismatch: expected={expected} actual={actual}")

with tarfile.open(archive, "r:gz") as bundle:
    members = bundle.getmembers()
    for member in members:
        destination = (project / member.name).resolve()
        if project not in destination.parents and destination != project:
            raise SystemExit(f"unsafe archive member: {member.name}")
        if not member.name.startswith("cache/avatarforcing_tts/"):
            raise SystemExit(f"unexpected archive member: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise SystemExit(f"unsupported archive member: {member.name}")
    bundle.extractall(project, members=members)
print(f"verified and extracted cache: {actual}")
PY
