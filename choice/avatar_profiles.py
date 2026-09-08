import copy
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = Path(
    os.environ.get(
        "LIVETALKING_AVATAR_PROFILE_REGISTRY",
        PROJECT_ROOT / "data" / "avatar_profiles.json",
    )
)


class AvatarProfileRegistry:
    """Loads avatar profiles and overlays cache-backed profiles from their manifest.

    A profile may declare ``choice.cache_manifest``.  In that case the generated
    cache metadata is the source of truth for voice, render and LatentSync
    compatibility fields.  This prevents a rebuilt cache from drifting away from
    the static registry or a runtime YAML file.
    """

    def __init__(self, registry_path: str | Path = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self._lock = Lock()
        self._dependency_paths: tuple[Path, ...] = ()
        self._signature: tuple[tuple[str, int], ...] = ()
        self._profiles: dict[str, dict[str, Any]] = {}

    def _resolve_path(self, value: str) -> str:
        if not value:
            return ""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())

    @staticmethod
    def _path_mtime(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except FileNotFoundError:
            return -1

    def _current_signature(self) -> tuple[tuple[str, int], ...]:
        paths = (self.registry_path, *self._dependency_paths)
        return tuple((str(path), self._path_mtime(path)) for path in paths)

    @staticmethod
    def _ready_item_metas(manifest: dict[str, Any]) -> list[dict[str, Any]]:
        metas = []
        for item in manifest.get("items", []):
            if item.get("status") != "ready":
                continue
            meta_path = Path(item.get("meta_path", ""))
            if not meta_path.is_file():
                raise FileNotFoundError(f"cache item metadata missing: {meta_path}")
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        if not metas:
            raise ValueError("cache manifest contains no ready item metadata")
        return metas

    @staticmethod
    def _common_cache_contract(metas: list[dict[str, Any]]) -> dict[str, Any]:
        fields = (
            "schema_version",
            "avatar_id",
            "model",
            "ref_image_path",
            "ref_image_sha1",
            "tts",
            "tts_ref_file",
            "tts_ref_file_sha1",
            "tts_ref_text",
            "tts_speed_factor",
            "tts_fragment_interval",
            "tts_streaming_mode",
            "render_pipeline",
            "latentsync_seed",
            "latentsync_enable_deepcache",
            "num_steps",
            "guidance_scale",
            "source_video_sha256",
            "fps",
        )
        contract = {field: metas[0].get(field) for field in fields}
        for index, meta in enumerate(metas[1:], start=2):
            for field, expected in contract.items():
                if meta.get(field) != expected:
                    raise ValueError(
                        "cache manifest mixes incompatible runtime contracts: "
                        f"item={index} field={field} first={expected!r} current={meta.get(field)!r}"
                    )
        return contract

    @staticmethod
    def _stream_profile_id(cache_root: Path, graph_id: str, cache_keys: set[str]) -> str:
        manifest_path = cache_root / graph_id / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        profile_ids = set()
        for raw_entries in payload.get("items", {}).values():
            entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
            for entry in entries:
                if (
                    entry.get("status") == "ready"
                    and entry.get("cache_key") in cache_keys
                    and entry.get("profile_id")
                ):
                    profile_ids.add(entry["profile_id"])
        if len(profile_ids) != 1:
            raise ValueError(
                "current cache items must resolve to exactly one render profile: "
                f"{sorted(profile_ids)}"
            )
        return next(iter(profile_ids))

    def _apply_cache_manifest(self, profile: dict[str, Any]) -> None:
        choice = profile.setdefault("choice", {})
        manifest_value = choice.get("cache_manifest")
        if not manifest_value:
            return

        manifest_path = Path(self._resolve_path(manifest_value))
        choice["cache_manifest_path"] = str(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"avatar cache manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metas = self._ready_item_metas(manifest)
        contract = self._common_cache_contract(metas)
        if contract["avatar_id"] != profile.get("avatar_id"):
            raise ValueError(
                f"cache avatar mismatch: profile={profile.get('avatar_id')!r} "
                f"cache={contract['avatar_id']!r}"
            )

        graph_id = manifest.get("graph_id") or manifest.get("stream_cache_tree_id")
        cache_root = Path(
            self._resolve_path(manifest.get("stream_cache_root", ""))
        )
        if not graph_id or not cache_root.is_dir():
            raise ValueError(
                f"invalid graph/cache root in avatar cache manifest: {manifest_path}"
            )
        cache_keys = {
            item.get("cache_key", "")
            for item in manifest.get("items", [])
            if item.get("status") == "ready" and item.get("cache_key")
        }
        render_profile_id = self._stream_profile_id(cache_root, graph_id, cache_keys)

        reference_audio = self._resolve_path(
            manifest.get("reference_audio") or contract["tts_ref_file"]
        )
        reference_image = self._resolve_path(
            manifest.get("reference_image") or contract["ref_image_path"]
        )
        voice = profile.setdefault("voice", {})
        voice.update(
            {
                "tts": contract["tts"],
                "profile": manifest.get("reference_profile") or voice.get("profile", ""),
                "ref_file": reference_audio,
                "ref_file_path": reference_audio,
                "ref_text": manifest.get("reference_text") or contract["tts_ref_text"],
                "speed_factor": contract["tts_speed_factor"],
                "fragment_interval": contract["tts_fragment_interval"],
                "streaming_mode": contract["tts_streaming_mode"],
            }
        )
        official_tts = (manifest.get("tts", {}).get("official_params") or {})
        voice.update(
            {
                "media_type": official_tts.get("media_type", "wav"),
                "text_lang": official_tts.get("text_lang", "zh"),
                "prompt_lang": official_tts.get("prompt_lang", "zh"),
                "split_method": official_tts.get("split_method", "cut5"),
            }
        )

        profile["reference_image"] = reference_image
        profile["reference_image_path"] = reference_image
        choice["graph_id"] = graph_id
        choice["render_profile_id"] = render_profile_id
        choice["legacy_cache_root"] = str(cache_root)
        choice["legacy_cache_root_path"] = str(cache_root)
        legacy = choice.setdefault("legacy", {})
        legacy.update(
            {
                "schema_version": contract["schema_version"],
                "render_pipeline": contract["render_pipeline"],
                "latentsync_seed": contract["latentsync_seed"],
                "latentsync_enable_deepcache": contract[
                    "latentsync_enable_deepcache"
                ],
                "num_steps": contract["num_steps"],
                "guidance_scale": contract["guidance_scale"],
                "source_video_sha256": contract["source_video_sha256"],
                "fps": contract["fps"],
            }
        )
        profile["cache_contract"] = {
            "manifest_path": str(manifest_path),
            "graph_id": graph_id,
            "ready_items": len(metas),
            "render_profile_id": render_profile_id,
            **contract,
        }

    def _load_if_needed(self) -> None:
        if not self.registry_path.is_file():
            self._profiles = {}
            self._signature = ()
            self._dependency_paths = ()
            return
        signature = self._current_signature()
        if self._profiles and signature == self._signature:
            return
        with self._lock:
            signature = self._current_signature()
            if self._profiles and signature == self._signature:
                return
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "avatar_profiles_v1":
                raise ValueError("unsupported avatar profile schema")
            profiles: dict[str, dict[str, Any]] = {}
            dependencies: list[Path] = []
            for raw in payload.get("profiles", []):
                profile = copy.deepcopy(raw)
                avatar_id = str(profile.get("avatar_id", "")).strip()
                if not avatar_id or avatar_id in profiles:
                    raise ValueError(f"invalid or duplicate avatar_id: {avatar_id!r}")
                for field in ("preview_image", "reference_image"):
                    profile[f"{field}_path"] = self._resolve_path(profile.get(field, ""))
                voice = profile.setdefault("voice", {})
                voice["ref_file_path"] = self._resolve_path(voice.get("ref_file", ""))
                choice = profile.setdefault("choice", {})
                if choice.get("legacy_cache_root"):
                    choice["legacy_cache_root_path"] = self._resolve_path(choice["legacy_cache_root"])
                if choice.get("cache_manifest"):
                    manifest_path = Path(self._resolve_path(choice["cache_manifest"]))
                    dependencies.append(manifest_path)
                    choice["cache_manifest_path"] = str(manifest_path)
                    self._apply_cache_manifest(profile)
                profiles[avatar_id] = profile
            self._profiles = profiles
            self._dependency_paths = tuple(dict.fromkeys(dependencies))
            self._signature = self._current_signature()

    def get(self, avatar_id: str, *, enabled_only: bool = False) -> Optional[dict[str, Any]]:
        self._load_if_needed()
        profile = self._profiles.get(avatar_id)
        if not profile or (enabled_only and not profile.get("enabled", False)):
            return None
        return copy.deepcopy(profile)

    def list(self, *, model: str = "", enabled_only: bool = True) -> list[dict[str, Any]]:
        self._load_if_needed()
        profiles = []
        for profile in self._profiles.values():
            if enabled_only and not profile.get("enabled", False):
                continue
            if model and profile.get("model") != model:
                continue
            profiles.append(copy.deepcopy(profile))
        return profiles


avatar_profiles = AvatarProfileRegistry()


def get_avatar_profile(avatar_id: str, *, enabled_only: bool = False) -> Optional[dict[str, Any]]:
    return avatar_profiles.get(avatar_id, enabled_only=enabled_only)


def apply_avatar_profile(opt, avatar_id: str) -> dict[str, Any]:
    profile = get_avatar_profile(avatar_id, enabled_only=True)
    if not profile:
        return {}
    opt.avatar_id = avatar_id
    if profile.get("model") and profile["model"] != getattr(opt, "model", ""):
        raise ValueError(
            f"avatar {avatar_id} requires model={profile['model']}, runtime={getattr(opt, 'model', '')}"
        )
    reference_image = Path(profile.get("reference_image_path", ""))
    if not reference_image.is_file():
        raise FileNotFoundError(f"avatar reference image missing: {reference_image}")
    voice = profile.get("voice", {})
    voice_path = Path(voice.get("ref_file_path", ""))
    if voice.get("ref_file_path") and not voice_path.is_file():
        raise FileNotFoundError(f"avatar voice reference missing: {voice_path}")
    if voice.get("tts"):
        opt.tts = voice["tts"]
    if voice.get("ref_file_path"):
        opt.REF_FILE = voice["ref_file_path"]
    opt.REF_TEXT = voice.get("ref_text", "")
    opt.voice_group = profile.get("voice_group", "")
    voice_runtime_map = {
        "speed_factor": "TTS_SPEED_FACTOR",
        "fragment_interval": "TTS_FRAGMENT_INTERVAL",
        "streaming_mode": "GPT_SOVITS_STREAMING_MODE",
        "media_type": "TTS_MEDIA_TYPE",
        "text_lang": "TTS_TEXT_LANG",
        "prompt_lang": "TTS_PROMPT_LANG",
        "split_method": "TTS_SPLIT_METHOD",
    }
    for field, option_name in voice_runtime_map.items():
        if voice.get(field) is not None:
            setattr(opt, option_name, voice[field])
    choice = profile.get("choice", {})
    opt.choice_graph_id = choice.get("graph_id", "daily_chat")
    opt.choice_tree_id = opt.choice_graph_id
    opt.render_profile_id = choice.get("render_profile_id", "")
    opt.choice_cache_manifest = choice.get("cache_manifest_path", "")
    opt.avatar_reference_image = profile.get("reference_image_path", "")
    opt.avatar_profile = profile
    return profile
