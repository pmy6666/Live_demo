import os
from collections import OrderedDict
from threading import Lock
from typing import Any

from choice.echomimicv3_cache import ChoiceEchoMimicV3CacheStore
from choice.legacy_latentsync_adapter import LegacyLatentSyncAdapter


class ProfileAwareMediaCache:
    def __init__(self, project_root):
        self.legacy_latentsync = LegacyLatentSyncAdapter()
        self.echomimicv3_v1 = ChoiceEchoMimicV3CacheStore(project_root)
        self.hot_cache_max_bytes = int(os.getenv("LIVETALKING_CHOICE_L1_BYTES", str(512 * 1024 * 1024)))
        self._hot_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._hot_cache_bytes = 0
        self._hot_cache_lock = Lock()

    def is_ready(self, avatar_session, graph_id: str, node: dict[str, Any]) -> bool:
        if self.legacy_latentsync.is_ready(avatar_session, graph_id, node):
            return True
        return bool(self.echomimicv3_v1.get_candidate_metas_by_node(graph_id, node["node_id"]))

    def lookup(self, avatar_session, graph_id: str, node: dict[str, Any], tts_text: str) -> dict[str, Any]:
        legacy = self.legacy_latentsync.lookup(avatar_session, graph_id, node, tts_text)
        if legacy.get("hit"):
            return legacy
        metas = self.echomimicv3_v1.get_candidate_metas_by_node(graph_id, node["node_id"])
        for meta in metas:
            if not self.echomimicv3_v1.is_compatible(avatar_session, node, tts_text, meta):
                continue
            segment = self.echomimicv3_v1.get(graph_id, meta.get("cache_key", ""))
            if segment:
                return {
                    "hit": True,
                    "level": "L2",
                    "mode": "echomimicv3_precomputed",
                    "asset_key": segment["meta"].get("cache_key", ""),
                    "profile_id": getattr(avatar_session.opt, "render_profile_id", ""),
                    "meta": segment["meta"],
                    "frames": segment["frames"],
                    "audio": segment["audio"],
                    "miss_reason": None,
                }
        return {
            "hit": False,
            "level": None,
            "mode": None,
            "asset_key": None,
            "miss_reason": legacy.get("miss_reason") or "video_asset_not_found",
        }

    def load_payload(self, asset: dict[str, Any]) -> dict[str, Any] | None:
        asset_key = str(asset.get("asset_key", ""))
        if asset_key:
            with self._hot_cache_lock:
                cached = self._hot_cache.get(asset_key)
                if cached is not None:
                    self._hot_cache.move_to_end(asset_key)
                    return cached
        if asset.get("payload_paths"):
            payload = self.legacy_latentsync.load_payload(asset)
        elif asset.get("frames") is not None and asset.get("audio") is not None:
            payload = {"frames": asset["frames"], "audio": asset["audio"]}
        else:
            payload = None
        if payload is None or not asset_key:
            return payload
        size = int(payload["frames"].nbytes + payload["audio"].nbytes)
        if size > self.hot_cache_max_bytes:
            return payload
        with self._hot_cache_lock:
            old = self._hot_cache.pop(asset_key, None)
            if old is not None:
                self._hot_cache_bytes -= int(old["frames"].nbytes + old["audio"].nbytes)
            self._hot_cache[asset_key] = payload
            self._hot_cache_bytes += size
            while self._hot_cache and self._hot_cache_bytes > self.hot_cache_max_bytes:
                _, evicted = self._hot_cache.popitem(last=False)
                self._hot_cache_bytes -= int(evicted["frames"].nbytes + evicted["audio"].nbytes)
        return payload

    def is_hot(self, asset_key: str) -> bool:
        with self._hot_cache_lock:
            return asset_key in self._hot_cache
