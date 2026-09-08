#!/usr/bin/env python3
"""Validated, fixed-parameter entry point for AvatarForcing LiveTalking."""

import json
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AVATAR_ID = "avatarforcing_male"
TTS_SERVER = os.environ.get("AVATARFORCING_TTS_SERVER", "http://127.0.0.1:9880")
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".runtime" / "torch"))
os.environ.setdefault(
    "LIVETALKING_AVATAR_PROFILE_REGISTRY",
    str(PROJECT_ROOT / "data" / "avatar_profiles_avatarforcing.json"),
)


def _require_file(path: Path, label: str, errors: list[str]):
    if not path.is_file():
        errors.append(f"{label} missing: {path}")


def _require_dir(path: Path, label: str, errors: list[str]):
    if not path.is_dir():
        errors.append(f"{label} directory missing: {path}")


def _load_default_profile(errors: list[str]) -> dict:
    registry_path = Path(os.environ["LIVETALKING_AVATAR_PROFILE_REGISTRY"])
    _require_file(registry_path, "Avatar profile registry", errors)
    if errors:
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    profile = next(
        (
            item
            for item in payload.get("profiles", [])
            if item.get("avatar_id") == DEFAULT_AVATAR_ID
        ),
        None,
    )
    if not profile:
        errors.append(
            f"Default avatar profile not found: {DEFAULT_AVATAR_ID} in {registry_path}"
        )
        return {}
    if profile.get("model") != "avatarforcing":
        errors.append(
            f"Profile {DEFAULT_AVATAR_ID} must declare model=avatarforcing; "
            f"found {profile.get('model')!r}"
        )
    if not profile.get("enabled"):
        errors.append(f"Profile {DEFAULT_AVATAR_ID} is disabled")
    for field, label in (
        ("reference_image", "Avatar reference image"),
        ("preview_image", "Avatar preview image"),
    ):
        value = profile.get(field, "")
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        _require_file(path, label, errors)
    voice_file = profile.get("voice", {}).get("ref_file", "")
    if voice_file:
        path = Path(voice_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        _require_file(path, "TTS reference audio", errors)
    return profile


def _check_tts_cache(voice_group: str, errors: list[str]) -> int:
    voice_group = str(voice_group).strip()
    if voice_group not in {"female", "male"}:
        errors.append(
            f"AvatarForcing voice_group must be male or female; found {voice_group!r}"
        )
        return 0

    graph_path = PROJECT_ROOT / "assets" / "daily_chat_playable_nodes.json"
    _require_file(graph_path, "Daily-chat playable graph", errors)
    if not graph_path.is_file():
        return 0
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = [node for node in graph.get("nodes", []) if node.get("enabled", True)]
    if not nodes:
        errors.append(f"Daily-chat playable graph contains no enabled nodes: {graph_path}")
        return 0

    cache_dir = PROJECT_ROOT / "cache" / "avatarforcing_tts" / voice_group
    _require_dir(cache_dir, f"AvatarForcing {voice_group} TTS cache", errors)
    if not cache_dir.is_dir():
        return 0

    ready = 0
    for node in nodes:
        node_id = str(node.get("node_id", "")).strip()
        path = cache_dir / f"{node_id}.npy"
        if not path.is_file():
            errors.append(
                f"missing AvatarForcing TTS cache: voice_group={voice_group} "
                f"node_id={node_id} path={path}"
            )
            continue
        try:
            audio = np.load(path, allow_pickle=False, mmap_mode="r")
            if (
                audio.ndim != 1
                or audio.size == 0
                or audio.dtype != np.float32
                or not np.isfinite(audio).all()
                or float(audio.min()) < -1.0
                or float(audio.max()) > 1.0
            ):
                raise ValueError(
                    "expected non-empty mono float32 PCM in [-1.0, 1.0]"
                )
        except Exception as exc:
            errors.append(
                f"invalid AvatarForcing TTS cache: voice_group={voice_group} "
                f"node_id={node_id} path={path}: {exc}"
            )
            continue
        ready += 1
    return ready


def validate_startup(require_cuda: bool = True):
    errors = []
    resources = {
        "Inference config": PROJECT_ROOT
        / "AvatarForcing"
        / "configs"
        / "inference.yaml",
        "Motion autoencoder checkpoint": PROJECT_ROOT
        / "AvatarForcing"
        / "pretrained_dir"
        / "motion_autoencoder.pth",
        "Flow transformer checkpoint": PROJECT_ROOT
        / "AvatarForcing"
        / "pretrained_dir"
        / "flow_transformer.pth",
    }
    for label, path in resources.items():
        _require_file(path, label, errors)

    wav2vec = (
        PROJECT_ROOT
        / "AvatarForcing"
        / "pretrained_dir"
        / "wav2vec2-base-960h"
    )
    _require_dir(wav2vec, "Wav2Vec2", errors)
    for name in ("config.json", "preprocessor_config.json", "vocab.json"):
        _require_file(wav2vec / name, f"Wav2Vec2 {name}", errors)
    if not any(
        (wav2vec / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        errors.append(
            f"Wav2Vec2 model weights missing in: {wav2vec} "
            "(expected model.safetensors or pytorch_model.bin)"
        )

    profile = _load_default_profile(errors)

    try:
        import torch

        if require_cuda and not torch.cuda.is_available():
            errors.append(
                "CUDA is unavailable. AvatarForcing Talking-only requires an "
                "NVIDIA GPU, driver, and CUDA-enabled PyTorch."
            )
    except ImportError as exc:
        errors.append(f"PyTorch import failed: {exc}")

    profile_voice_group = str(profile.get("voice_group", "")).strip()
    if profile_voice_group not in {"female", "male"}:
        errors.append(
            f"Profile {DEFAULT_AVATAR_ID} must declare voice_group=male or female; "
            f"found {profile_voice_group!r}"
        )
    ready_tts_caches = {
        voice_group: _check_tts_cache(voice_group, errors)
        for voice_group in ("female", "male")
    }
    capture_root = Path(
        os.environ.get(
            "LIVETALKING_CAMERA_CAPTURE_ROOT",
            "/tmp/livetalking-avatar-captures",
        )
    )
    try:
        capture_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        errors.append(f"Camera capture temp directory unavailable: {capture_root}: {exc}")
    if errors:
        raise RuntimeError(
            "AvatarForcing startup validation failed:\n- " + "\n- ".join(errors)
        )

    print("AvatarForcing startup validation passed")
    print(f"  project: {PROJECT_ROOT}")
    print(f"  avatar: {DEFAULT_AVATAR_ID}")
    for voice_group, ready in ready_tts_caches.items():
        print(f"  offline TTS cache: voice_group={voice_group} ready={ready}")
    print(f"  camera capture temp: {capture_root}")
    print(f"  config: {resources['Inference config']}")
    print(f"  mae checkpoint: {resources['Motion autoencoder checkpoint']}")
    print(f"  flow checkpoint: {resources['Flow transformer checkpoint']}")
    print(f"  wav2vec: {wav2vec}")
    print(
        "  Talking-only: seed=20 nfe=10 a_cfg_scale=2.0 "
        "u_cfg_scale=0.0 pad_ratio=1.0 kv_cache=True"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Start AvatarForcing with a fixed inference contract."
    )
    parser.add_argument(
        "--change",
        choices=("True", "False", "true", "false"),
        default="True",
        help=(
            "Compatibility fallback for direct engine calls. LiveTalking "
            "always pastes back static assets and never pastes back camera selfies."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the final-demo files and CUDA environment, then exit.",
    )
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="With --validate-only, skip the CUDA availability check.",
    )
    cli_args = parser.parse_args()
    if cli_args.files_only and not cli_args.validate_only:
        parser.error("--files-only requires --validate-only")
    os.chdir(PROJECT_ROOT)
    validate_startup(require_cuda=not cli_args.files_only)
    if cli_args.validate_only:
        return
    sys.argv = [
        str(Path(__file__).resolve()),
        "--model",
        "avatarforcing",
        "--avatar_id",
        DEFAULT_AVATAR_ID,
        "--transport",
        "webrtc",
        "--listenport",
        "8010",
        "--max_session",
        "1",
        "--fps",
        "25",
        "--choice_config",
        "",
        "--change",
        cli_args.change,
        "--TTS_SERVER",
        TTS_SERVER,
    ]
    mp.set_start_method("spawn")
    import app

    app.main()


if __name__ == "__main__":
    main()
