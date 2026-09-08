#!/usr/bin/env python3
"""Publish female caches and generate male AvatarForcing TTS caches.

The output format follows docs/AvatarForcing/AvatarForcing男女双音色TTS缓存简化技术报告.md:

    cache/avatarforcing_tts/<voice_group>/<node_id>.npy

Each NPY contains mono, 16 kHz, float32 PCM in [-1.0, 1.0].
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import requests
import resampy
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = PROJECT_ROOT / "assets" / "daily_chat_playable_nodes.json"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cache" / "avatarforcing_tts"
DEFAULT_FEMALE_MANIFEST = PROJECT_ROOT / "Final_catch" / "female" / "manifest.json"
DEFAULT_MALE_REFERENCE = PROJECT_ROOT / "bilibili_downloads" / "male.mp3"
DEFAULT_REFERENCE_TEXTS = PROJECT_ROOT / "docs" / "notes" / "content.txt"
DEFAULT_TTS_PARAMS = (
    PROJECT_ROOT / "gpt_sovits_official_materials" / "current_tts_params.json"
)
SAFE_NODE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def status(message: str) -> None:
    print(f"[avatarforcing-tts-cache] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, path)


def validate_audio(audio: np.ndarray, label: str) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"{label}: expected non-empty mono audio, got shape={audio.shape}")
    if audio.dtype != np.float32:
        raise ValueError(f"{label}: expected float32, got {audio.dtype}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{label}: audio contains NaN or infinity")
    minimum = float(audio.min())
    maximum = float(audio.max())
    if minimum < -1.0 or maximum > 1.0:
        raise ValueError(f"{label}: PCM range is [{minimum}, {maximum}], expected [-1, 1]")
    return audio


def load_existing_audio(path: Path) -> np.ndarray:
    return validate_audio(np.load(path, allow_pickle=False), str(path))


def save_audio(path: Path, audio: np.ndarray) -> None:
    audio = validate_audio(np.asarray(audio, dtype=np.float32), str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with staging.open("wb") as output:
            np.save(output, audio, allow_pickle=False)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def load_nodes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "daily_chat_graph_v2":
        raise ValueError(f"Unsupported graph schema in {path}")
    nodes = [node for node in payload.get("nodes", []) if node.get("enabled", True)]
    if not nodes:
        raise ValueError(f"No enabled nodes found in {path}")
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("node_id", "")).strip()
        text = str(node.get("tts_text", "")).strip()
        if not SAFE_NODE_ID.fullmatch(node_id):
            raise ValueError(f"Unsafe or empty node_id: {node_id!r}")
        if node_id in seen:
            raise ValueError(f"Duplicate node_id: {node_id}")
        if not text:
            raise ValueError(f"Empty tts_text: {node_id}")
        seen.add(node_id)
    expected = payload.get("statistics", {}).get("playable_nodes")
    if expected is not None and len(nodes) != int(expected):
        raise ValueError(f"Graph node count mismatch: loaded={len(nodes)} expected={expected}")
    return nodes


def read_named_transcript(path: Path, name: str) -> str:
    wanted = name.casefold()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        separators = [position for position in (line.find(":"), line.find("：")) if position >= 0]
        if not separators:
            continue
        split_at = min(separators)
        key = line[:split_at].strip().casefold()
        if key != wanted:
            continue
        value = line[split_at + 1 :].strip()
        # content.txt may append a gender marker after an ASCII-quoted transcript
        # (for example: male: "..." M). Only the quoted speech is the prompt.
        if value.startswith('"') and value.find('"', 1) >= 0:
            return value[1 : value.find('"', 1)].strip()
        return value.strip('"“”')
    raise ValueError(f"Transcript {name!r} was not found in {path}")


def publish_female(args: argparse.Namespace, nodes: list[dict]) -> int:
    source_manifest = json.loads(args.female_manifest.read_text(encoding="utf-8"))
    source_items = {
        str(item.get("source_node_id", "")): item
        for item in source_manifest.get("items", [])
    }
    expected_ids = {node["node_id"] for node in nodes}
    missing = sorted(expected_ids - source_items.keys())
    extra = sorted(source_items.keys() - expected_ids)
    if missing or extra:
        raise ValueError(
            f"Female manifest/graph mismatch: missing={missing[:10]} extra={extra[:10]}"
        )

    output_dir = args.cache_root / "female"
    records = []
    for index, node in enumerate(nodes, start=1):
        node_id = node["node_id"]
        source_path = Path(source_items[node_id].get("audio_npy_path", ""))
        if not source_path.is_file():
            raise FileNotFoundError(f"Female source cache missing: {node_id}: {source_path}")
        audio = load_existing_audio(source_path)
        target = output_dir / f"{node_id}.npy"
        if target.exists() and not args.force:
            existing = load_existing_audio(target)
            if existing.shape == audio.shape and sha256_file(target) == sha256_file(source_path):
                action = "kept"
            else:
                raise RuntimeError(
                    f"Existing female cache differs: {target}; rerun with --force to replace it"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                shutil.copyfile(source_path, staging)
                load_existing_audio(staging)
                os.replace(staging, target)
            finally:
                staging.unlink(missing_ok=True)
            action = "published"
        records.append(
            {
                "node_id": node_id,
                "tts_text": node["tts_text"],
                "path": str(target),
                "samples": int(audio.size),
                "duration_seconds": round(audio.size / 16000.0, 3),
                "sha256": sha256_file(target),
            }
        )
        status(f"female {index}/{len(nodes)} {action}: {node_id}")

    manifest = {
        "schema_version": "avatarforcing_tts_cache_v1",
        "voice_group": "female",
        "sample_rate": 16000,
        "channels": 1,
        "dtype": "float32",
        "source_manifest": str(args.female_manifest),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "items": records,
    }
    atomic_write_json(args.cache_root / "female_manifest.json", manifest)
    status(f"female complete: {len(records)} files in {output_dir}")
    return 0


def decode_tts_wav(content: bytes, node_id: str) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(io.BytesIO(content), dtype="float32", always_2d=True)
    except Exception as exc:
        raise RuntimeError(f"GPT-SoVITS returned invalid WAV for {node_id}: {exc}") from exc
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    if sample_rate != 16000:
        audio = resampy.resample(audio, sample_rate, 16000)
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return validate_audio(audio, node_id)


def generate_male(args: argparse.Namespace, nodes: list[dict]) -> int:
    prompt_text = read_named_transcript(args.reference_texts, "male")
    params = json.loads(args.tts_params.read_text(encoding="utf-8"))
    output_dir = args.cache_root / "male"
    manifest_path = args.cache_root / "male_manifest.json"
    reference_hash = sha256_file(args.male_reference)
    records: list[dict] = []

    server = args.tts_server.rstrip("/")
    try:
        response = requests.get(f"{server}/docs", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"GPT-SoVITS is not ready at {server}; start it before running this command"
        ) from exc

    for index, node in enumerate(nodes, start=1):
        node_id = node["node_id"]
        target = output_dir / f"{node_id}.npy"
        if target.exists() and not args.force:
            audio = load_existing_audio(target)
            action = "kept"
        else:
            payload = {
                "text": node["tts_text"],
                "text_lang": str(params.get("text_lang", "zh")),
                "ref_audio_path": str(args.male_reference),
                "prompt_text": prompt_text,
                "prompt_lang": str(params.get("prompt_lang", "zh")),
                "text_split_method": str(params.get("split_method", "cut5")),
                "media_type": "wav",
                "streaming_mode": int(params.get("streaming_mode", 0)),
                "speed_factor": float(params.get("speed_factor", 1.08)),
                "fragment_interval": float(params.get("fragment_interval", 0.1)),
            }
            status(f"male {index}/{len(nodes)} generating: {node_id}")
            response = requests.post(
                f"{server}/tts",
                json=payload,
                timeout=args.timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"GPT-SoVITS failed for {node_id}: "
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            audio = decode_tts_wav(response.content, node_id)
            save_audio(target, audio)
            action = "generated"

        records.append(
            {
                "node_id": node_id,
                "tts_text": node["tts_text"],
                "path": str(target),
                "samples": int(audio.size),
                "duration_seconds": round(audio.size / 16000.0, 3),
                "sha256": sha256_file(target),
                "status": action,
            }
        )
        manifest = {
            "schema_version": "avatarforcing_tts_cache_v1",
            "voice_group": "male",
            "sample_rate": 16000,
            "channels": 1,
            "dtype": "float32",
            "tts_server": server,
            "reference_audio": str(args.male_reference),
            "reference_audio_sha256": reference_hash,
            "reference_text": prompt_text,
            "tts_params": params,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "complete": len(records) == len(nodes),
            "items": records,
        }
        atomic_write_json(manifest_path, manifest)
        status(f"male {index}/{len(nodes)} {action}: {node_id}")

    status(f"male complete: {len(records)} files in {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simple male/female NPY TTS caches for AvatarForcing."
    )
    parser.add_argument(
        "action",
        choices=("publish-female", "generate-male", "verify"),
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--female-manifest", type=Path, default=DEFAULT_FEMALE_MANIFEST)
    parser.add_argument("--male-reference", type=Path, default=DEFAULT_MALE_REFERENCE)
    parser.add_argument("--reference-texts", type=Path, default=DEFAULT_REFERENCE_TEXTS)
    parser.add_argument("--tts-params", type=Path, default=DEFAULT_TTS_PARAMS)
    parser.add_argument("--tts-server", default="http://127.0.0.1:9880")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def verify(args: argparse.Namespace, nodes: list[dict]) -> int:
    failed = False
    for voice_group in ("female", "male"):
        missing = []
        invalid = []
        for node in nodes:
            path = args.cache_root / voice_group / f"{node['node_id']}.npy"
            if not path.is_file():
                missing.append(node["node_id"])
                continue
            try:
                load_existing_audio(path)
            except Exception as exc:
                invalid.append(f"{node['node_id']}: {exc}")
        status(
            f"verify {voice_group}: ready={len(nodes) - len(missing) - len(invalid)} "
            f"missing={len(missing)} invalid={len(invalid)}"
        )
        for node_id in missing:
            print(f"missing {voice_group} cache: {node_id}")
        for message in invalid:
            print(f"invalid {voice_group} cache: {message}")
        failed = failed or bool(missing or invalid)
    return 1 if failed else 0


def main() -> int:
    args = parse_args()
    for name in ("graph",):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    nodes = load_nodes(args.graph)
    status(f"loaded {len(nodes)} enabled Choice nodes from {args.graph}")

    if args.action == "publish-female":
        if not args.female_manifest.is_file():
            raise FileNotFoundError(f"female manifest not found: {args.female_manifest}")
        return publish_female(args, nodes)
    if args.action == "generate-male":
        for name in ("male_reference", "reference_texts", "tts_params"):
            path = getattr(args, name)
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")
        return generate_male(args, nodes)
    return verify(args, nodes)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted; completed files are preserved for the next run.", file=sys.stderr)
        raise SystemExit(130)
