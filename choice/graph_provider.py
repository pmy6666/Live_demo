import copy
import json
from pathlib import Path
from threading import Lock
from typing import Any


ALLOWED_EDGE_TYPES = {
    "continue",
    "deepen",
    "practical",
    "soft_bridge",
    "topic_transfer",
    "restart",
}


class GraphValidationError(ValueError):
    pass


class GraphProvider:
    """Loads and validates versioned choice graphs as immutable snapshots."""

    def __init__(self, graph_root: str | Path):
        self.graph_root = Path(graph_root)
        self.project_root = self.graph_root.resolve().parents[1]
        self._cache: dict[str, tuple[str, int, dict[str, Any]]] = {}
        self._lock = Lock()

    def load_graph(self, graph_id: str) -> dict[str, Any]:
        path = self._resolve_graph_path(graph_id)
        stat = path.stat()
        with self._lock:
            cached = self._cache.get(graph_id)
            if cached and cached[0] == str(path) and cached[1] == stat.st_mtime_ns:
                return cached[2]
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload = self._normalize(payload)
            self._validate(payload, graph_id)
            payload["node_map"] = {node["node_id"]: node for node in payload["nodes"]}
            outgoing: dict[str, list[dict[str, Any]]] = {}
            for edge in payload.get("edges", []):
                outgoing.setdefault(edge["from_node_id"], []).append(edge)
            payload["outgoing_edges"] = outgoing
            payload["source_path"] = str(path)
            self._cache[graph_id] = (str(path), stat.st_mtime_ns, payload)
            return payload

    def _resolve_graph_path(self, graph_id: str) -> Path:
        """Use the published playable graph as daily_chat's single source of truth."""
        if graph_id == "daily_chat":
            playable_graph = self.project_root / "assets" / "daily_chat_playable_nodes.json"
            if playable_graph.is_file():
                return playable_graph
        return self.graph_root / f"{graph_id}.json"

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("schema_version") != "daily_chat_graph_v2":
            return payload

        normalized = copy.deepcopy(payload)
        virtual_root = normalized.get("virtual_root", {})
        if not isinstance(virtual_root, dict):
            raise GraphValidationError("daily_chat virtual_root must be an object")

        normalized["source_schema_version"] = normalized["schema_version"]
        normalized["schema_version"] = "choice_graph_v2"
        normalized["virtual_root"] = virtual_root.get("node_id")
        normalized["entry_pool"] = list(virtual_root.get("entry_pool", []))

        aliases_by_node: dict[str, list[str]] = {}
        for alias, canonical in normalized.get("legacy_node_aliases", {}).items():
            aliases_by_node.setdefault(canonical, []).append(alias)
        for node in normalized.get("nodes", []):
            aliases = list(node.get("legacy_aliases", []))
            aliases.extend(aliases_by_node.get(node.get("node_id", ""), []))
            legacy_node_id = (node.get("legacy_cache") or {}).get("node_id")
            if legacy_node_id and legacy_node_id != node.get("node_id"):
                aliases.append(legacy_node_id)
            node["legacy_aliases"] = list(dict.fromkeys(aliases))

        related_topics = {
            topic["topic_id"]: list(topic.get("transfer_topic_ids", []))
            for topic in normalized.get("topics", [])
            if topic.get("topic_id")
        }
        transition_policy = normalized.setdefault("transition_policy", {})
        transition_policy["related_topics"] = related_topics
        return normalized

    def snapshot(self, graph_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.load_graph(graph_id))

    def get_node(self, graph_id: str, node_id: str) -> dict[str, Any]:
        graph = self.load_graph(graph_id)
        try:
            return graph["node_map"][node_id]
        except KeyError as exc:
            raise KeyError(f"choice graph node not found: {graph_id}/{node_id}") from exc

    @staticmethod
    def _validate(payload: dict[str, Any], expected_graph_id: str) -> None:
        if payload.get("schema_version") != "choice_graph_v2":
            raise GraphValidationError("graph schema_version must be choice_graph_v2")
        if payload.get("graph_id") != expected_graph_id:
            raise GraphValidationError("graph_id does not match filename")
        if not payload.get("graph_version"):
            raise GraphValidationError("graph_version is required")
        if not payload.get("virtual_root"):
            raise GraphValidationError("virtual_root is required")
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise GraphValidationError("nodes must be an array")
        node_ids: set[str] = set()
        for node in nodes:
            node_id = node.get("node_id")
            if not node_id or node_id in node_ids:
                raise GraphValidationError(f"invalid or duplicate node_id: {node_id!r}")
            node_ids.add(node_id)
            if not node.get("topic_id") or not isinstance(node.get("round_in_topic"), int):
                raise GraphValidationError(f"node topic/round missing: {node_id}")
            if not node.get("question_text") or not node.get("answer_text"):
                raise GraphValidationError(f"node text missing: {node_id}")
        entry_pool = payload.get("entry_pool", [])
        if len(entry_pool) != 20 or len(set(entry_pool)) != 20:
            raise GraphValidationError("entry_pool must contain exactly 20 unique nodes")
        for node_id in entry_pool:
            node = next((item for item in nodes if item["node_id"] == node_id), None)
            if not node or node.get("round_in_topic") != 1 or not node.get("enabled", True):
                raise GraphValidationError(f"invalid entry node: {node_id}")
        edge_ids: set[str] = set()
        for edge in payload.get("edges", []):
            edge_id = edge.get("edge_id")
            if not edge_id or edge_id in edge_ids:
                raise GraphValidationError(f"invalid or duplicate edge_id: {edge_id!r}")
            edge_ids.add(edge_id)
            if edge.get("from_node_id") not in node_ids or edge.get("to_node_id") not in node_ids:
                raise GraphValidationError(f"edge references missing node: {edge_id}")
            if edge.get("edge_type") not in ALLOWED_EDGE_TYPES:
                raise GraphValidationError(f"invalid edge_type: {edge_id}")
