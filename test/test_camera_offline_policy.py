import asyncio
import json
import types
import unittest

from server import routes
from server.session_manager import SessionManager


class _OfflineSession:
    def __init__(self):
        self.opt = types.SimpleNamespace(conversation_mode="offline_choice_only")
        self.flush_count = 0
        self.text_count = 0
        self.cleanup_count = 0

    def flush_talk(self, reason=""):
        self.flush_count += 1
        return self.flush_count

    def put_msg_txt(self, *_args, **_kwargs):
        self.text_count += 1

    def cleanup(self):
        self.cleanup_count += 1


class _Request:
    def __init__(self, payload, app=None):
        self._payload = payload
        self.app = app or {}

    async def json(self):
        return self._payload


class CameraOfflinePolicyTest(unittest.TestCase):
    def setUp(self):
        routes.session_manager.sessions.clear()
        routes.session_manager.created_at.clear()

    def tearDown(self):
        routes.session_manager.sessions.clear()
        routes.session_manager.created_at.clear()

    def test_human_text_is_rejected_before_tts_or_llm(self):
        session = _OfflineSession()
        routes.session_manager.add_session("camera-session", session)
        response = asyncio.run(
            routes.human(
                _Request(
                    {
                        "sessionid": "camera-session",
                        "type": "echo",
                        "text": "must not play",
                        "interrupt": True,
                    }
                )
            )
        )
        payload = json.loads(response.text)
        self.assertEqual(payload["msg"], "offline_choice_only")
        self.assertEqual(session.text_count, 0)
        self.assertEqual(session.flush_count, 0)

    def test_remove_session_runs_cleanup_callback_once(self):
        manager = SessionManager()
        manager.sessions.clear()
        manager.created_at.clear()
        session = _OfflineSession()
        session._session_cleanup = session.cleanup
        manager.add_session("camera-session", session)
        manager.remove_session("camera-session")
        manager.remove_session("camera-session")
        self.assertEqual(session.flush_count, 1)
        self.assertEqual(session.cleanup_count, 1)

    def test_camera_avatar_is_always_advertised_for_avatarforcing_runtime(self):
        response = asyncio.run(
            routes.list_avatars(
                _Request(
                    {},
                    app={"runtime_model": "avatarforcing"},
                )
            )
        )
        avatar_ids = [
            item["id"]
            for item in json.loads(response.text)["data"]["avatars"]
        ]
        self.assertIn("avatarforcing_camera", avatar_ids)


if __name__ == "__main__":
    unittest.main()
