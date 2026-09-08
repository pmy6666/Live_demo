import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np

from choice.avatar_profiles import apply_avatar_profile
from utils.logger import logger


def _sha1_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LegacyLatentSyncAdapter:
    """Read-only adapter for finalized female LatentSync frame bundles."""

    def __init__(self):
        self._lock = Lock()
        self._manifest_cache: dict[str, tuple[int, dict[str, Any]]] = {}

    def _manifest(self, cache_root: str, graph_id: str) -> dict[str, Any]:
        path = Path(cache_root) / graph_id / "manifest.json"
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return {}
        cache_key = str(path)
        with self._lock:
            cached = self._manifest_cache.get(cache_key)
            if cached and cached[0] == mtime:
                return cached[1]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("load legacy LatentSync manifest failed: %s", path)
                return {}
            self._manifest_cache[cache_key] = (mtime, payload)
            return payload

    def _profile_context(self, avatar_session) -> tuple[Optional[dict[str, Any]], str]:
        avatar_id = getattr(avatar_session.opt, "avatar_id", "")
        profile = apply_avatar_profile(avatar_session.opt, avatar_id)
        if not profile:
            return None, "avatar_profile_missing"
        choice = profile.get("choice", {})
        if not choice.get("legacy_cache_root_path"):
            return None, "legacy_cache_not_configured"
        runtime_profile = getattr(avatar_session.opt, "render_profile_id", "")
        if runtime_profile != choice.get("render_profile_id"):
            return None, "render_profile_mismatch"
        return profile, ""

    def is_ready(self, avatar_session, graph_id: str, node: dict[str, Any]) -> bool:
        profile, _ = self._profile_context(avatar_session)
        if not profile:
            return False
        choice = profile["choice"]
        manifest = self._manifest(choice["legacy_cache_root_path"], graph_id)
        for legacy_id in node.get("legacy_aliases", []) + [node["node_id"]]:
            entries = manifest.get("items", {}).get(legacy_id, [])
            entries = entries if isinstance(entries, list) else [entries]
            if any(
                entry.get("status") == "ready"
                and entry.get("profile_id") == choice.get("render_profile_id")
                for entry in entries
            ):
                return True
        return False

    def lookup(
        self,
        avatar_session,
        graph_id: str,
        node: dict[str, Any],
        tts_text: str,
    ) -> dict[str, Any]:
        profile, reason = self._profile_context(avatar_session)
        if not profile:
            return {"hit": False, "miss_reason": reason}
        choice = profile["choice"]
        manifest = self._manifest(choice["legacy_cache_root_path"], graph_id)
        if not manifest:
            return {"hit": False, "miss_reason": "legacy_manifest_missing"}
        if manifest.get("schema_version") != choice.get("legacy", {}).get("schema_version"):
            return {"hit": False, "miss_reason": "legacy_schema_mismatch"}

        aliases = list(node.get("legacy_aliases", [])) + [node["node_id"]]
        for legacy_id in aliases:
            entries = manifest.get("items", {}).get(legacy_id, [])
            entries = entries if isinstance(entries, list) else [entries]
            for entry in entries:
                if entry.get("status") != "ready":
                    continue
                if entry.get("profile_id") != choice.get("render_profile_id"):
                    continue
                result = self._load_and_validate(
                    avatar_session, profile, graph_id, legacy_id, entry, node, tts_text
                )
                if result.get("hit"):
                    return result
                reason = result.get("miss_reason", reason)
        return {"hit": False, "miss_reason": reason or "legacy_asset_not_ready"}

    def _load_and_validate(
        self,
        avatar_session,
        profile: dict[str, Any],
        graph_id: str,
        legacy_id: str,
        entry: dict[str, Any],
        node: dict[str, Any],
        tts_text: str,
    ) -> dict[str, Any]:
        cache_key = entry.get("cache_key", "")
        cache_dir = Path(profile["choice"]["legacy_cache_root_path"]) / graph_id / cache_key
        required = {
            "meta": cache_dir / "meta.json",
            "audio": cache_dir / "audio.npy",
            "frames": cache_dir / "frames.npz",
            "preview": cache_dir / "preview.jpg",
        }
        if not cache_key or not all(path.is_file() for path in required.values()):
            return {"hit": False, "miss_reason": "legacy_payload_incomplete"}
        try:
            meta = json.loads(required["meta"].read_text(encoding="utf-8"))
        except Exception:
            return {"hit": False, "miss_reason": "legacy_meta_invalid"}

        opt = avatar_session.opt
        legacy = profile["choice"].get("legacy", {})
        avatar = getattr(avatar_session, "avatar", None)
        checks = {
            "schema_version": legacy.get("schema_version"),
            "tree_id": graph_id,
            "node_id": legacy_id,
            "answer_text": tts_text,
            "avatar_id": profile["avatar_id"],
            "model": profile.get("model"),
            "tts": profile.get("voice", {}).get("tts"),
            "ref_image_sha1": _sha1_file(getattr(avatar, "ref_image_path", "")),
            "tts_ref_file_sha1": _sha1_file(getattr(opt, "REF_FILE", "")),
            "tts_ref_text": getattr(opt, "REF_TEXT", ""),
            "render_pipeline": legacy.get("render_pipeline"),
            "latentsync_seed": legacy.get("latentsync_seed"),
            "latentsync_enable_deepcache": legacy.get("latentsync_enable_deepcache"),
            "num_steps": legacy.get("num_steps"),
            "guidance_scale": legacy.get("guidance_scale"),
            "source_video_sha256": legacy.get("source_video_sha256"),
            "fps": legacy.get("fps"),
        }
        for field, expected in checks.items():
            if expected is not None and meta.get(field) != expected:
                logger.info(
                    "legacy LatentSync miss node=%s legacy=%s field=%s cached=%r runtime=%r",
                    node["node_id"],
                    legacy_id,
                    field,
                    meta.get(field),
                    expected,
                )
                return {"hit": False, "miss_reason": f"legacy_{field}_mismatch"}
        return {
            "hit": True,
            "level": "L2",
            "mode": "legacy_latentsync_predecoded",
            "asset_key": cache_key,
            "profile_id": entry.get("profile_id"),
            "legacy_node_id": legacy_id,
            "meta": meta,
            "payload_paths": {key: str(path) for key, path in required.items()},
            "miss_reason": None,
        }

    @staticmethod
    def load_payload(asset: dict[str, Any]) -> Optional[dict[str, np.ndarray]]:
        paths = asset.get("payload_paths", {})
        try:
            audio = np.load(paths["audio"], allow_pickle=False).astype(np.float32, copy=False)
            with np.load(paths["frames"], allow_pickle=False) as payload:
                frames = payload["frames"]
            meta = asset.get("meta", {})
            if audio.dtype != np.float32 or audio.ndim != 1 or audio.size <= 0:
                return None
            if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] <= 0 or frames.shape[-1] != 3:
                return None
            if int(meta.get("frames", -1)) != int(frames.shape[0]):
                return None
            return {"frames": frames, "audio": audio}
        except Exception:
            logger.exception("load legacy LatentSync payload failed: %s", paths)
            return None
