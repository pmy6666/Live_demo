import hashlib
import os
import random
from typing import Any, Callable, Optional


CacheReadyFn = Optional[Callable[[str], bool]]


class TransitionPolicy:
    def __init__(self, server_secret: str | None = None):
        self.server_secret = server_secret or os.getenv(
            "LIVETALKING_CHOICE_SEED_SECRET", "livetalking-daily-chat-v1"
        )

    def session_seed(self, session_id: str, graph_version: str) -> int:
        raw = f"{self.server_secret}\0{session_id}\0{graph_version}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")

    @staticmethod
    def _choice(
        node: dict[str, Any],
        edge_type: str,
        text: str | None = None,
        choice_id: str | None = None,
    ) -> dict[str, Any]:
        node_id = node["node_id"]
        return {
            "choice_id": choice_id or node_id,
            "choice_text": text or node.get("display_text") or node["question_text"],
            "child_node_id": node_id,
            "edge_type": edge_type,
        }

    def sample_entries(
        self,
        graph: dict[str, Any],
        random_seed: int,
        cache_ready: CacheReadyFn = None,
        sample_index: int = 0,
    ) -> list[dict[str, Any]]:
        rng = random.Random(random_seed + sample_index * 104729)
        nodes = [graph["node_map"][node_id] for node_id in graph["entry_pool"]]
        categories: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            categories.setdefault(node.get("category", "other"), []).append(node)
        category_names = list(categories)
        rng.shuffle(category_names)
        selected: list[dict[str, Any]] = []
        for category in category_names:
            bucket = categories[category]
            weights = [1.1 if cache_ready and cache_ready(node["node_id"]) else 1.0 for node in bucket]
            selected.append(rng.choices(bucket, weights=weights, k=1)[0])
            if len(selected) == 3:
                break
        if len(selected) < 3:
            remaining = [node for node in nodes if node not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: 3 - len(selected)])
        rng.shuffle(selected)
        return [self._choice(node, "entry") for node in selected]

    def choices_after_node(
        self,
        graph: dict[str, Any],
        node: dict[str, Any],
        topic_history: list[str],
        random_seed: int,
        state_version: int,
        cache_ready: CacheReadyFn = None,
    ) -> list[dict[str, Any]]:
        rng = random.Random(random_seed + state_version * 1009)
        choices: list[dict[str, Any]] = []
        for edge in graph["outgoing_edges"].get(node["node_id"], []):
            if not edge.get("enabled", True):
                continue
            target = graph["node_map"][edge["to_node_id"]]
            if not target.get("enabled", True):
                continue
            choices.append(
                self._choice(
                    target,
                    edge["edge_type"],
                    edge.get("choice_text"),
                    edge.get("choice_id") or target["node_id"],
                )
            )

        # A valid full graph already supplies three reviewed outgoing edges.
        # Related topic entries are only a defensive fallback for partial graphs.
        related = graph.get("transition_policy", {}).get("related_topics", {}).get(node["topic_id"], [])
        recent_topics = set(topic_history[-2:])
        transfer_nodes = []
        for topic_id in related:
            candidate = next(
                (
                    graph["node_map"][entry_id]
                    for entry_id in graph["entry_pool"]
                    if graph["node_map"][entry_id]["topic_id"] == topic_id
                ),
                None,
            )
            if candidate and candidate["topic_id"] != node["topic_id"]:
                transfer_nodes.append(candidate)
        transfer_nodes.sort(key=lambda item: item["topic_id"] in recent_topics)
        for target in transfer_nodes:
            if len(choices) >= 3:
                break
            if any(item["child_node_id"] == target["node_id"] for item in choices):
                continue
            choices.append(self._choice(target, "topic_transfer", f"聊聊：{target['question_text']}"))

        if len(choices) < 3:
            fallback = [
                graph["node_map"][entry_id]
                for entry_id in graph["entry_pool"]
                if graph["node_map"][entry_id]["topic_id"] != node["topic_id"]
                and not any(item["child_node_id"] == entry_id for item in choices)
            ]
            rng.shuffle(fallback)
            for target in fallback[: 3 - len(choices)]:
                choices.append(self._choice(target, "topic_transfer", f"聊聊：{target['question_text']}"))

        for choice in choices:
            choice["cache_ready"] = bool(cache_ready and cache_ready(choice["child_node_id"]))
        return choices[:3]
