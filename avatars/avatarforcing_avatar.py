"""AvatarForcing Talking-only backend for LiveTalking.

The model is loaded once by ``app.py``. Each session collects a complete cached
utterance and submits it to the official Talking-only inference path. Playback
starts after three seconds of generated video are buffered, while later model
blocks continue to be inferred and queued with the original 16 kHz PCM.
"""

import importlib
import hashlib
import math
import os
import queue
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from avatars.base_avatar import BaseAvatar
from choice.avatar_profiles import get_avatar_profile
from registry import register
from utils.logger import logger


SEED = 20
NFE = 10
A_CFG_SCALE = 2.0
U_CFG_SCALE = 0.0
PAD_RATIO = 1.0
USE_KV_CACHE = True
MAX_AUDIO_SECONDS = 30
STREAM_BUFFER_SECONDS = 3.0
STREAM_BUFFER_FRAMES = math.ceil(STREAM_BUFFER_SECONDS * 25)
IDLE_BREATH_CYCLE_SECONDS = 4.8
IDLE_BREATH_SCALE = 0.0025
IDLE_BREATH_LIFT_PX = 0.6


@dataclass
class AvatarForcingAvatarData:
    avatar_id: str
    ref_image_path: str
    idle_frame: np.ndarray
    paste_back: bool


@dataclass(frozen=True)
class AvatarForcingPreparedReference:
    tensor: torch.Tensor
    face_frame_bgr: np.ndarray
    original_frame_bgr: np.ndarray
    crop_meta: dict


@dataclass(frozen=True)
class AvatarForcingAudioJob:
    audio: np.ndarray
    start_event: dict
    end_event: dict
    token: int


class AvatarReferenceValidationError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _absolute(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _make_video_compatible_canvas(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if height % 2 == 0 and width % 2 == 0:
        return frame
    return cv2.copyMakeBorder(
        frame,
        0,
        height % 2,
        0,
        width % 2,
        borderType=cv2.BORDER_REPLICATE,
    )


class AvatarForcingEngine:
    """Single-GPU, serialized wrapper around the official InferenceAgent."""

    def __init__(self, opt):
        self.repo_path = _absolute(opt.avatarforcing_repo)
        self.config_path = _absolute(opt.avatarforcing_config)
        self.mae_ckpt_path = _absolute(opt.avatarforcing_mae_ckpt)
        self.ckpt_path = _absolute(opt.avatarforcing_ckpt)
        self.wav2vec_path = _absolute(opt.avatarforcing_wav2vec)
        # Compatibility fallback for direct engine callers. LiveTalking
        # sessions use AvatarForcingAvatarData.paste_back so static and camera
        # avatars can select different output modes in the same process.
        self.change = bool(getattr(opt, "change", True))
        self._validate_paths()

        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        inference = importlib.import_module("inference")
        talking_only = importlib.import_module("inference_talking_only")
        self._preprocess_talking_only = talking_only.preprocess_talking_only
        self._seed_everything = inference.seed_everything

        self._seed_everything(SEED)
        infer_config = OmegaConf.load(self.config_path)
        runtime = OmegaConf.create(
            {
                "infer_config": self.config_path,
                "mae_ckpt_path": self.mae_ckpt_path,
                "ckpt_path": self.ckpt_path,
                "wav2vec_model_path": self.wav2vec_path,
                "seed": SEED,
                "nfe": NFE,
                "a_cfg_scale": A_CFG_SCALE,
                "pad_ratio": PAD_RATIO,
                "rank": 0,
                "ngpus": 1,
            }
        )
        self.model_opt = OmegaConf.merge(infer_config, runtime)
        if int(self.model_opt.fps) != 25 or int(self.model_opt.sampling_rate) != 16000:
            raise ValueError(
                "AvatarForcing config must use fps=25 and sampling_rate=16000"
            )

        logger.info("AvatarForcing loading InferenceAgent (one global model)")
        self.agent = inference.InferenceAgent(self.model_opt)
        self._lock = Lock()
        self._reference_lock = Lock()
        self._avatar_refs = {}
        logger.info(
            "AvatarForcing Talking-only ready: seed=%d nfe=%d a_cfg_scale=%.1f "
            "u_cfg_scale=%.1f pad_ratio=%.1f kv_cache=%s "
            "default_paste_back=%s config=%s "
            "mae_ckpt=%s ckpt=%s wav2vec=%s",
            SEED,
            NFE,
            A_CFG_SCALE,
            U_CFG_SCALE,
            PAD_RATIO,
            USE_KV_CACHE,
            self.change,
            self.config_path,
            self.mae_ckpt_path,
            self.ckpt_path,
            self.wav2vec_path,
        )

    def _validate_paths(self):
        required_files = {
            "inference config": self.config_path,
            "motion autoencoder checkpoint": self.mae_ckpt_path,
            "flow transformer checkpoint": self.ckpt_path,
            "official inference module": os.path.join(self.repo_path, "inference.py"),
            "Talking-only module": os.path.join(
                self.repo_path, "inference_talking_only.py"
            ),
        }
        missing = [
            f"{label}: {path}"
            for label, path in required_files.items()
            if not os.path.isfile(path)
        ]
        if not os.path.isdir(self.wav2vec_path):
            missing.append(f"Wav2Vec2 directory: {self.wav2vec_path}")
        if missing:
            raise FileNotFoundError(
                "AvatarForcing required resources are missing:\n- "
                + "\n- ".join(missing)
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for AvatarForcing Talking-only inference."
            )

    def prepare_avatar(self, ref_image_path: str):
        ref_image_path = _absolute(ref_image_path)
        with self._reference_lock:
            cached = self._avatar_refs.get(ref_image_path)
        if cached is not None:
            return cached
        processor = self.agent.data_processor
        source_frame = cv2.imread(ref_image_path)
        if source_frame is None:
            raise FileNotFoundError(
                f"AvatarForcing reference image is unreadable: {ref_image_path}"
            )
        source_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
        source_height, source_width = source_rgb.shape[:2]
        scale = 360.0 / source_height
        detector_frame = cv2.resize(
            source_rgb,
            dsize=(0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
        detected = processor.fa.face_detector.detect_from_image(detector_frame)
        confident = [box for box in detected if float(box[4]) > 0.95]
        if not detected:
            raise AvatarReferenceValidationError("no_face_detected")
        if not confident:
            raise AvatarReferenceValidationError("face_confidence_too_low")
        if len(confident) > 1:
            raise AvatarReferenceValidationError("multiple_faces_detected")
        x1, y1, x2, y2, _ = confident[0]
        face_width = (x2 - x1) / scale
        face_height = (y2 - y1) / scale
        if min(face_width, face_height) < max(80, min(source_width, source_height) * 0.12):
            raise AvatarReferenceValidationError("face_too_small")
        try:
            face, crop_meta = processor.preprocess_face(
                ref_image_path,
                pad_ratio=PAD_RATIO,
                return_crop_meta=True,
            )
            tensor = processor.transform(image=face)["image"].unsqueeze(0)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"AvatarForcing could not detect a face with confidence > 0.95: "
                f"{ref_image_path}"
            ) from exc
        if source_frame.shape[:2] != (
            crop_meta["original_height"],
            crop_meta["original_width"],
        ):
            raise ValueError(
                "AvatarForcing reference dimensions changed during preprocessing: "
                f"{ref_image_path}"
            )
        original_frame = _make_video_compatible_canvas(source_frame)
        crop_meta = dict(crop_meta)
        crop_meta["canvas_width"] = original_frame.shape[1]
        crop_meta["canvas_height"] = original_frame.shape[0]
        reference = AvatarForcingPreparedReference(
            tensor=tensor,
            face_frame_bgr=cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
            original_frame_bgr=original_frame,
            crop_meta=crop_meta,
        )
        with self._reference_lock:
            self._avatar_refs[ref_image_path] = reference
        logger.info(
            "AvatarForcing reference prepared: path_hash=%s face=%dx%d "
            "canvas=%dx%d crop=(%d,%d)-(%d,%d)",
            hashlib.sha256(ref_image_path.encode()).hexdigest()[:12],
            reference.face_frame_bgr.shape[1],
            reference.face_frame_bgr.shape[0],
            crop_meta["canvas_width"],
            crop_meta["canvas_height"],
            crop_meta["crop_x1"],
            crop_meta["crop_y1"],
            crop_meta["crop_x2"],
            crop_meta["crop_y2"],
        )
        return reference

    def release_avatar(self, ref_image_path: str) -> bool:
        """Forget a dynamic reference without touching the global model weights."""
        ref_image_path = _absolute(ref_image_path)
        with self._reference_lock:
            removed = self._avatar_refs.pop(ref_image_path, None)
        if removed is not None:
            path_hash = hashlib.sha256(ref_image_path.encode()).hexdigest()[:12]
            logger.info("AvatarForcing reference released: path_hash=%s", path_hash)
        return removed is not None

    @staticmethod
    def _tensor_to_bgr_frames(
        d_hat: torch.Tensor,
        value_min=None,
        value_max=None,
    ) -> list[np.ndarray]:
        # Streaming uses the first playback buffer as a stable normalization
        # reference so later blocks do not flicker due to per-block min/max.
        video = d_hat.permute(0, 2, 3, 1).detach().clamp(-1, 1).cpu()
        minimum = video.min() if value_min is None else value_min
        maximum = video.max() if value_max is None else value_max
        value_range = maximum - minimum
        if float(value_range) == 0.0:
            video = torch.zeros_like(video, dtype=torch.uint8)
        else:
            video = (
                (video - minimum) / value_range * 255
            ).clamp(0, 255).to(torch.uint8)
        rgb = video.numpy()
        return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb]

    @staticmethod
    def _paste_frames(
        frames: list[np.ndarray],
        reference: AvatarForcingPreparedReference,
    ) -> list[np.ndarray]:
        meta = reference.crop_meta
        x1, y1 = meta["crop_x1"], meta["crop_y1"]
        x2, y2 = meta["crop_x2"], meta["crop_y2"]
        crop_size = (meta["crop_width"], meta["crop_height"])
        output = []
        for frame in frames:
            canvas = reference.original_frame_bgr.copy()
            canvas[y1:y2, x1:x2] = cv2.resize(
                frame,
                crop_size,
                interpolation=cv2.INTER_LINEAR,
            )
            output.append(canvas)
        return output

    def generate_frames(
        self,
        ref_image_path: str,
        audio: np.ndarray,
        paste_back: bool | None = None,
    ) -> list[np.ndarray]:
        paste_back = self.change if paste_back is None else bool(paste_back)
        ref_image_path = _absolute(ref_image_path)
        self.prepare_avatar(ref_image_path)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size > MAX_AUDIO_SECONDS * 16000:
            raise ValueError(
                f"AvatarForcing input is {audio.size / 16000:.2f}s; "
                f"the Talking-only limit is {MAX_AUDIO_SECONDS}s. Split the text "
                "into shorter utterances."
            )

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = handle.name
            sf.write(temp_path, audio, 16000, subtype="PCM_16")

            # Keep the official helper as the preprocessing source of truth.
            # Replace its reference result with the startup-validated cached
            # tensor so inference uses exactly that validated avatar crop.
            data = self._preprocess_talking_only(
                self.agent, ref_image_path, temp_path, PAD_RATIO
            )
            reference = self._avatar_refs[ref_image_path]
            data["avatar_ref"] = reference.tensor

            with self._lock:
                self._seed_everything(SEED)
                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self.agent.G.inference(
                        data=data,
                        a_cfg_scale=A_CFG_SCALE,
                        u_cfg_scale=U_CFG_SCALE,
                        nfe=NFE,
                        seed=SEED,
                        use_kv_cache=USE_KV_CACHE,
                    )
                torch.cuda.synchronize()
                frames = self._tensor_to_bgr_frames(output["d_hat"])
                if paste_back:
                    frames = self._paste_frames(frames, reference)
                logger.info(
                    "AvatarForcing inference complete: frames=%d duration=%.3fs "
                    "output=%dx%d paste_back=%s elapsed=%.3fs",
                    len(frames),
                    len(frames) / 25.0,
                    frames[0].shape[1] if frames else 0,
                    frames[0].shape[0] if frames else 0,
                    paste_back,
                    time.perf_counter() - started,
                )
                return frames
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def generate_frames_stream(
        self,
        ref_image_path: str,
        audio: np.ndarray,
        frame_callback,
        paste_back: bool | None = None,
    ) -> int:
        """Generate blocks and publish playback after a three-second buffer."""
        paste_back = self.change if paste_back is None else bool(paste_back)
        ref_image_path = _absolute(ref_image_path)
        self.prepare_avatar(ref_image_path)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size > MAX_AUDIO_SECONDS * 16000:
            raise ValueError(
                f"AvatarForcing input is {audio.size / 16000:.2f}s; "
                f"the Talking-only limit is {MAX_AUDIO_SECONDS}s. Split the text "
                "into shorter utterances."
            )

        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = handle.name
            sf.write(temp_path, audio, 16000, subtype="PCM_16")
            data = self._preprocess_talking_only(
                self.agent, ref_image_path, temp_path, PAD_RATIO
            )
            reference = self._avatar_refs[ref_image_path]
            data["avatar_ref"] = reference.tensor

            buffered_tensors = []
            buffered_frames = 0
            emitted_frames = 0
            normalization_range = None

            with self._lock:
                self._seed_everything(SEED)
                started = time.perf_counter()

                def publish_block(d_hat, start_frame, total_frames):
                    nonlocal buffered_frames
                    nonlocal emitted_frames
                    nonlocal normalization_range

                    cpu_block = d_hat.detach().clamp(-1, 1).cpu()
                    if normalization_range is None:
                        buffered_tensors.append(cpu_block)
                        buffered_frames += cpu_block.shape[0]
                        if (
                            buffered_frames < STREAM_BUFFER_FRAMES
                            and buffered_frames < total_frames
                        ):
                            return
                        cpu_block = torch.cat(buffered_tensors, dim=0)
                        buffered_tensors.clear()
                        normalization_range = (
                            cpu_block.min(),
                            cpu_block.max(),
                        )
                        start_frame = 0
                        logger.info(
                            "AvatarForcing stream playback ready: buffered=%d "
                            "content=%.2fs total_frames=%d elapsed=%.3fs",
                            buffered_frames,
                            buffered_frames / 25.0,
                            total_frames,
                            time.perf_counter() - started,
                        )

                    frames = self._tensor_to_bgr_frames(
                        cpu_block,
                        value_min=normalization_range[0],
                        value_max=normalization_range[1],
                    )
                    if paste_back:
                        frames = self._paste_frames(frames, reference)
                    frame_callback(frames, start_frame, total_frames)
                    emitted_frames += len(frames)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = self.agent.G.inference_stream(
                        data=data,
                        block_callback=publish_block,
                        a_cfg_scale=A_CFG_SCALE,
                        u_cfg_scale=U_CFG_SCALE,
                        nfe=NFE,
                        seed=SEED,
                        use_kv_cache=USE_KV_CACHE,
                    )
                torch.cuda.synchronize()
                total_frames = int(result["frames"])
                logger.info(
                    "AvatarForcing streaming inference complete: "
                    "frames=%d emitted=%d duration=%.3fs paste_back=%s elapsed=%.3fs",
                    total_frames,
                    emitted_frames,
                    total_frames / 25.0,
                    paste_back,
                    time.perf_counter() - started,
                )
                return emitted_frames
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass


def load_model(opt):
    return AvatarForcingEngine(opt)


def load_avatar(avatar_id):
    profile = get_avatar_profile(avatar_id, enabled_only=True)
    if not profile:
        raise ValueError(f"AvatarForcing avatar profile is not enabled: {avatar_id}")
    if profile.get("model") != "avatarforcing":
        raise ValueError(
            f"Avatar profile {avatar_id} must declare model=avatarforcing"
        )
    ref_image_path = profile.get("reference_image_path", "")
    frame = cv2.imread(ref_image_path)
    if frame is None:
        raise FileNotFoundError(
            f"AvatarForcing reference image is unreadable: {ref_image_path}"
        )
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    return AvatarForcingAvatarData(
        avatar_id=avatar_id,
        ref_image_path=ref_image_path,
        idle_frame=frame,
        paste_back=True,
    )


def load_avatar_from_reference(
    avatar_id: str,
    ref_image_path: str,
) -> AvatarForcingAvatarData:
    """Load a backend-trusted capture path without registering it globally."""
    ref_image_path = _absolute(ref_image_path)
    frame = cv2.imread(ref_image_path)
    if frame is None:
        raise FileNotFoundError(
            f"AvatarForcing reference image is unreadable: {ref_image_path}"
        )
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    return AvatarForcingAvatarData(
        avatar_id=avatar_id,
        ref_image_path=ref_image_path,
        idle_frame=frame,
        paste_back=False,
    )


def warm_up(opt, model: AvatarForcingEngine, avatar: AvatarForcingAvatarData):
    # Face preprocessing validates the reference and removes first-utterance
    # detector latency.  Diffusion generation is deliberately not run here.
    reference = model.prepare_avatar(avatar.ref_image_path)
    if avatar.paste_back:
        avatar.idle_frame = reference.original_frame_bgr.copy()
    else:
        avatar.idle_frame = reference.face_frame_bgr.copy()
    logger.info(
        "AvatarForcing warm-up complete: avatar=%s paste_back=%s "
        "(reference preprocessing only)",
        avatar.avatar_id,
        avatar.paste_back,
    )


@register("avatar", "avatarforcing")
class AvatarForcingReal(BaseAvatar):
    def __init__(
        self,
        opt,
        model: AvatarForcingEngine,
        avatar: AvatarForcingAvatarData,
    ):
        super().__init__(opt)
        self.engine = model
        self.avatar = avatar
        self.frame_list_cycle = [avatar.idle_frame]
        self._audio_jobs = queue.Queue()
        self._playback_frames = queue.Queue(maxsize=256)
        self._pending_audio = []
        self._pending_start_event = {}
        self._pending_lock = Lock()
        self._idle_animation_started = time.perf_counter()
        self._talk_transition_pending = Event()

    def _animated_idle_frame(self, now: float) -> np.ndarray:
        frame = self.avatar.idle_frame
        if self._talk_transition_pending.is_set():
            return frame
        height, width = frame.shape[:2]
        cycle_position = (
            (now - self._idle_animation_started) % IDLE_BREATH_CYCLE_SECONDS
        ) / IDLE_BREATH_CYCLE_SECONDS
        # Starts and ends at the exact reference pose. The cosine envelope
        # avoids a velocity jump at the inhale/exhale turning points.
        breath = 0.5 - 0.5 * np.cos(2.0 * np.pi * cycle_position)
        scale = 1.0 + IDLE_BREATH_SCALE * breath
        anchor = (width * 0.5, height * 0.38)
        transform = cv2.getRotationMatrix2D(anchor, 0.0, scale)
        transform[1, 2] -= IDLE_BREATH_LIFT_PX * breath
        return cv2.warpAffine(
            frame,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def put_audio_frame(self, audio_chunk: np.ndarray, datainfo: dict = None):
        datainfo = dict(datainfo or {})
        playback_id = datainfo.get("playback_id")
        if playback_id is not None and not self.playback_controller.is_current(
            int(playback_id)
        ):
            return
        self._capture_choice_audio(audio_chunk, datainfo)
        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        status = datainfo.get("status")

        with self._pending_lock:
            if status == "start":
                self._pending_audio = []
                self._pending_start_event = dict(datainfo)
            if chunk.size:
                self._pending_audio.append(chunk.copy())
            if status == "end":
                if self._pending_audio:
                    audio = np.concatenate(self._pending_audio).astype(
                        np.float32, copy=False
                    )
                    duration = audio.size / self.sample_rate
                    if duration > MAX_AUDIO_SECONDS:
                        logger.error(
                            "AvatarForcing rejected %.2fs utterance; maximum is %ds. "
                            "Split the input text and retry.",
                            duration,
                            MAX_AUDIO_SECONDS,
                        )
                    else:
                        token = self.current_playback_token()
                        self._talk_transition_pending.set()
                        self._audio_jobs.put(
                            AvatarForcingAudioJob(
                                audio=audio,
                                start_event=dict(self._pending_start_event),
                                end_event=dict(datainfo),
                                token=token,
                            )
                        )
                        logger.info(
                            "AvatarForcing utterance queued: samples=%d duration=%.2fs",
                            audio.size,
                            duration,
                        )
                self._pending_audio = []
                self._pending_start_event = {}

    def flush_talk(self, reason: str = "interrupt"):
        playback_id = super().flush_talk(reason=reason)
        self._clear_queue(self._audio_jobs)
        self._clear_queue(self._playback_frames)
        with self._pending_lock:
            self._pending_audio = []
            self._pending_start_event = {}
        self._talk_transition_pending.clear()
        self._idle_animation_started = time.perf_counter()
        return playback_id

    def _put_playback_frame(self, payload, token: int) -> bool:
        while token == self.current_playback_token():
            try:
                self._playback_frames.put(payload, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _enqueue_playback_block(
        self,
        frames: list[np.ndarray],
        start_frame: int,
        total_frames: int,
        job: AvatarForcingAudioJob,
    ):
        for local_index, frame in enumerate(frames):
            if job.token != self.current_playback_token():
                return
            frame_index = start_frame + local_index
            chunks = []
            for sub_index in range(2):
                chunk_index = frame_index * 2 + sub_index
                start = chunk_index * self.chunk
                event = {}
                if chunk_index == 0:
                    event.update(job.start_event)
                    event["status"] = "start"
                if chunk_index == total_frames * 2 - 1:
                    event.update(job.end_event)
                    event["status"] = "end"
                audio_chunk = np.zeros(self.chunk, dtype=np.float32)
                available = job.audio[start : start + self.chunk]
                audio_chunk[:available.size] = available
                chunks.append(
                    (
                        audio_chunk,
                        event,
                    )
                )
            if not self._put_playback_frame(
                (frame, chunks, job.token), job.token
            ):
                return

    def _generation_loop(self, quit_event: Event):
        while not quit_event.is_set():
            try:
                job = self._audio_jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job.token != self.current_playback_token():
                continue
            try:
                generation_started = time.perf_counter()
                playback_started = False

                def enqueue_block(frames, start_frame, total_frames):
                    nonlocal playback_started
                    if job.token != self.current_playback_token():
                        return
                    if not playback_started:
                        playback_started = True
                        expected_samples = total_frames * 2 * self.chunk
                        logger.info(
                            "AvatarForcing playback started while inference is "
                            "running: buffered_frames=%d first_frame_delay=%.3fs "
                            "original_samples=%d aligned_samples=%d delta_ms=%.1f",
                            len(frames),
                            time.perf_counter() - generation_started,
                            job.audio.size,
                            expected_samples,
                            abs(expected_samples - job.audio.size) / 16.0,
                        )
                    self._enqueue_playback_block(
                        frames,
                        start_frame,
                        total_frames,
                        job,
                    )

                emitted_frames = self.engine.generate_frames_stream(
                    self.avatar.ref_image_path,
                    job.audio,
                    enqueue_block,
                    paste_back=self.avatar.paste_back,
                )
                if (
                    not emitted_frames
                    or job.token != self.current_playback_token()
                ):
                    self._talk_transition_pending.clear()
            except Exception:
                self._talk_transition_pending.clear()
                logger.exception(
                    "AvatarForcing generation failed; audio was not played to "
                    "avoid unsynchronized or non-AvatarForcing output"
                )

    def render(self, quit_event):
        self.quit_event = quit_event
        self.init_customindex()
        if hasattr(self, "tts"):
            self.tts.render(quit_event)
        else:
            logger.info(
                "AvatarForcing render using offline TTS cache; "
                "realtime TTS worker skipped"
            )
        self.output.start()

        generation_quit = Event()
        generation_thread = Thread(
            target=self._generation_loop,
            args=(generation_quit,),
            name=f"avatarforcing-{self.sessionid}",
        )
        generation_thread.start()

        frame_interval = 1.0 / self.opt.fps
        silent_chunk = np.zeros(self.chunk, dtype=np.float32)
        try:
            while not quit_event.is_set():
                started = time.perf_counter()
                playback_id = self.current_playback_token()
                try:
                    frame, audio_chunks, token = self._playback_frames.get_nowait()
                    if token != self.current_playback_token():
                        continue
                    playback_id = token
                    self.speaking = True
                except queue.Empty:
                    if self.speaking:
                        self._talk_transition_pending.clear()
                        self._idle_animation_started = started
                    frame = self._animated_idle_frame(started)
                    audio_chunks = [(silent_chunk, {}), (silent_chunk, {})]
                    self.speaking = False

                output_frame = frame.copy()
                self.output.push_video_frame(
                    output_frame, playback_id=playback_id
                )
                self.record_video_data(output_frame)
                for audio_chunk, event in audio_chunks:
                    event = dict(event or {})
                    event.setdefault("playback_id", playback_id)
                    pcm = (
                        np.clip(audio_chunk, -1.0, 1.0) * 32767
                    ).astype(np.int16)
                    self.output.push_audio_frame(
                        pcm, event, playback_id=playback_id
                    )
                    self.record_audio_data(pcm)

                buffer_size = self.output.get_buffer_size()
                sleep_time = max(
                    0.0, frame_interval - (time.perf_counter() - started)
                )
                if buffer_size >= 5:
                    sleep_time += frame_interval * buffer_size * 0.8
                time.sleep(sleep_time)
        finally:
            generation_quit.set()
            generation_thread.join()
            self.output.stop()
            logger.info("AvatarForcing render thread stopped")
