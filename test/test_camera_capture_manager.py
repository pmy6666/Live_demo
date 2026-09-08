import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from server.camera_capture_manager import (
    CameraCaptureError,
    CameraCaptureManager,
)


class _Prepared:
    face_frame_bgr = np.zeros((512, 512, 3), dtype=np.uint8)


def _jpeg(width=800, height=600):
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def _png(width=800, height=600):
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return encoded.tobytes()


class CameraCaptureManagerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.now = [1000.0]
        self.released = []
        self.manager = CameraCaptureManager(
            root=self.temp_dir.name,
            validator=lambda _path: _Prepared(),
            reference_releaser=self.released.append,
            ttl_seconds=300,
            clock=lambda: self.now[0],
        )

    def tearDown(self):
        self.manager.close()
        self.temp_dir.cleanup()

    def test_create_normalizes_to_private_server_path(self):
        record = self.manager.create(_jpeg(), "female")
        self.assertEqual(record.state, "ready")
        self.assertEqual(record.voice_group, "female")
        self.assertEqual((record.width, record.height), (800, 600))
        self.assertRegex(record.capture_id, r"^[0-9a-f]{32}$")
        self.assertEqual(Path(record.image_path).name, "reference.jpg")
        self.assertTrue(Path(record.image_path).is_file())
        self.assertNotIn("camera.jpg", record.image_path)

    def test_uploaded_png_uses_same_no_pasteback_reference_contract(self):
        record = self.manager.create(_png(), "male")
        self.assertEqual(record.state, "ready")
        self.assertEqual(record.voice_group, "male")
        self.assertEqual(Path(record.image_path).name, "reference.jpg")
        self.assertTrue(Path(record.image_path).is_file())

    def test_capture_can_only_be_claimed_once(self):
        record = self.manager.create(_jpeg(), "male")
        claimed = self.manager.claim(record.capture_id, "session-a")
        self.assertEqual(claimed.owner_sessionid, "session-a")
        with self.assertRaisesRegex(CameraCaptureError, "capture_already_claimed"):
            self.manager.claim(record.capture_id, "session-b")
        with self.assertRaisesRegex(CameraCaptureError, "capture_already_claimed"):
            self.manager.cancel_ready(record.capture_id)

    def test_destroy_releases_reference_and_deletes_directory(self):
        record = self.manager.create(_jpeg(), "female")
        capture_dir = Path(record.image_path).parent
        self.manager.claim(record.capture_id, "session-a")
        self.assertTrue(
            self.manager.destroy(record.capture_id, owner_sessionid="session-a")
        )
        self.assertFalse(capture_dir.exists())
        self.assertIn(record.image_path, self.released)

    def test_expired_ready_capture_is_cleaned(self):
        record = self.manager.create(_jpeg(), "female")
        self.now[0] = record.expires_at + 1
        self.assertEqual(self.manager.cleanup_expired(), 1)
        self.assertIsNone(self.manager.get(record.capture_id))
        self.assertFalse(Path(record.image_path).exists())

    def test_invalid_voice_and_content_are_rejected(self):
        with self.assertRaisesRegex(CameraCaptureError, "invalid_voice_group"):
            self.manager.create(_jpeg(), "unknown")
        with self.assertRaisesRegex(CameraCaptureError, "unsupported_image_format"):
            self.manager.create(b"not an image", "male")
        with self.assertRaisesRegex(CameraCaptureError, "face_too_small"):
            self.manager.create(_jpeg(width=320, height=320), "male")

    def test_validator_error_code_is_preserved(self):
        class FaceError(ValueError):
            code = "multiple_faces_detected"

        manager = CameraCaptureManager(
            root=Path(self.temp_dir.name) / "second",
            validator=lambda _path: (_ for _ in ()).throw(FaceError()),
        )
        with self.assertRaisesRegex(CameraCaptureError, "multiple_faces_detected"):
            manager.create(_jpeg(), "female")
        self.assertEqual(list((Path(self.temp_dir.name) / "second").iterdir()), [])

    def test_upload_rate_limit(self):
        for _ in range(6):
            self.manager.check_rate_limit("client-a")
        with self.assertRaisesRegex(CameraCaptureError, "capture_rate_limited"):
            self.manager.check_rate_limit("client-a")
        self.now[0] += 61
        self.manager.check_rate_limit("client-a")


if __name__ == "__main__":
    unittest.main()
