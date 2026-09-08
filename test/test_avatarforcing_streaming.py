import queue
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from avatars.avatarforcing_avatar import (
    AvatarForcingAvatarData,
    AvatarForcingAudioJob,
    AvatarForcingEngine,
    AvatarForcingReal,
    warm_up,
)


class FakeStreamingModel:
    def inference_stream(self, data, block_callback, **kwargs):
        total_frames = 94
        for start_frame, frame_count in (
            (0, 50),
            (50, 10),
            (60, 10),
            (70, 10),
            (80, 10),
            (90, 4),
        ):
            block = torch.linspace(
                -1,
                1,
                frame_count * 3 * 2 * 2,
                dtype=torch.float32,
            ).reshape(frame_count, 3, 2, 2)
            block_callback(block, start_frame, total_frames)
        return {"frames": total_frames}


class AvatarForcingStreamingTests(unittest.TestCase):
    def test_engine_publishes_after_three_second_buffer_then_per_block(self):
        engine = AvatarForcingEngine.__new__(AvatarForcingEngine)
        engine.agent = SimpleNamespace(G=FakeStreamingModel())
        engine._lock = Lock()
        engine._seed_everything = lambda seed: None
        engine._preprocess_talking_only = (
            lambda agent, image, audio, ratio: {
                "avatar_ref": torch.zeros(1, 3, 2, 2)
            }
        )
        engine.change = True
        engine._paste_frames = lambda *_args: self.fail(
            "camera session must not paste generated frames back"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = str(Path(temp_dir) / "avatar.png")
            reference = SimpleNamespace(tensor=torch.zeros(1, 3, 2, 2))
            engine._avatar_refs = {image_path: reference}
            engine.prepare_avatar = lambda path: reference
            published = []

            with patch("torch.autocast", return_value=nullcontext()), patch(
                "torch.cuda.synchronize"
            ):
                emitted = engine.generate_frames_stream(
                    image_path,
                    np.zeros(94 * 640, dtype=np.float32),
                    lambda frames, start, total: published.append(
                        (start, len(frames), total)
                    ),
                    paste_back=False,
                )

        self.assertEqual(94, emitted)
        self.assertEqual(
            [(0, 80, 94), (80, 10, 94), (90, 4, 94)],
            published,
        )

    def test_warm_up_uses_avatar_level_paste_back_policy(self):
        reference = SimpleNamespace(
            original_frame_bgr=np.zeros((6, 10, 3), dtype=np.uint8),
            face_frame_bgr=np.ones((4, 4, 3), dtype=np.uint8),
        )
        engine = SimpleNamespace(prepare_avatar=lambda _path: reference)
        opt = SimpleNamespace()

        static_avatar = AvatarForcingAvatarData(
            avatar_id="avatarforcing_male",
            ref_image_path="/static.png",
            idle_frame=np.empty((1, 1, 3), dtype=np.uint8),
            paste_back=True,
        )
        camera_avatar = AvatarForcingAvatarData(
            avatar_id="avatarforcing_camera",
            ref_image_path="/camera.jpg",
            idle_frame=np.empty((1, 1, 3), dtype=np.uint8),
            paste_back=False,
        )

        warm_up(opt, engine, static_avatar)
        warm_up(opt, engine, camera_avatar)

        self.assertEqual((6, 10, 3), static_avatar.idle_frame.shape)
        self.assertEqual((4, 4, 3), camera_avatar.idle_frame.shape)

    def test_playback_blocks_keep_global_audio_and_event_positions(self):
        avatar = AvatarForcingReal.__new__(AvatarForcingReal)
        avatar.chunk = 320
        avatar._playback_frames = queue.Queue()
        avatar.current_playback_token = lambda: 7
        audio = np.arange(4 * 2 * avatar.chunk, dtype=np.float32)
        job = AvatarForcingAudioJob(
            audio=audio,
            start_event={"node_id": "start-node"},
            end_event={"node_id": "end-node"},
            token=7,
        )
        frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]

        avatar._enqueue_playback_block(frames[:3], 0, 4, job)
        avatar._enqueue_playback_block(frames[3:], 3, 4, job)
        payloads = [avatar._playback_frames.get_nowait() for _ in range(4)]

        chunks = [chunk for _, frame_chunks, _ in payloads for chunk in frame_chunks]
        np.testing.assert_array_equal(audio[:320], chunks[0][0])
        np.testing.assert_array_equal(audio[-320:], chunks[-1][0])
        self.assertEqual("start", chunks[0][1]["status"])
        self.assertNotIn("status", chunks[1][1])
        self.assertEqual("end", chunks[-1][1]["status"])
        self.assertEqual("start-node", chunks[0][1]["node_id"])
        self.assertEqual("end-node", chunks[-1][1]["node_id"])


if __name__ == "__main__":
    unittest.main()
