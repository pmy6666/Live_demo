import json
import unittest
from pathlib import Path

import start_avatarforcing
from choice.avatar_profiles import avatar_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AvatarForcingReleaseTests(unittest.TestCase):
    def test_release_profile_registry_is_isolated(self):
        profiles = avatar_profiles.list(enabled_only=True)
        self.assertEqual(
            ["avatarforcing_female", "avatarforcing_male"],
            [profile["avatar_id"] for profile in profiles],
        )
        self.assertTrue(all(profile["model"] == "avatarforcing" for profile in profiles))

    def test_both_voice_caches_cover_every_enabled_node(self):
        graph_path = PROJECT_ROOT / "assets" / "daily_chat_playable_nodes.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        expected = {
            node["node_id"]
            for node in graph["nodes"]
            if node.get("enabled", True)
        }
        self.assertEqual(260, len(expected))
        for voice_group in ("female", "male"):
            actual = {
                path.stem
                for path in (
                    PROJECT_ROOT / "cache" / "avatarforcing_tts" / voice_group
                ).glob("*.npy")
            }
            self.assertEqual(expected, actual, voice_group)

    def test_every_inventory_entry_exists(self):
        inventory = PROJECT_ROOT / "release" / "avatarforcing" / "runtime-files.txt"
        missing = []
        for line in inventory.read_text(encoding="utf-8").splitlines():
            relative = line.strip()
            if not relative or relative.startswith("#"):
                continue
            if not (PROJECT_ROOT / relative).exists():
                missing.append(relative)
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
