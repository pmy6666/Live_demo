"""Short-lived, single-use camera reference images for AvatarForcing sessions."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable, Optional

import cv2
import numpy as np


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_DECODED_PIXELS = 12_000_000
MIN_SHORT_EDGE = 480
MAX_LONG_EDGE = 1920
CAPTURE_TTL_SECONDS = 300
VOICE_GROUPS = frozenset({"female", "male"})


class CameraCaptureError(ValueError):
    """A stable, browser-displayable camera capture error."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class CameraCaptureRecord:
    capture_id: str
    image_path: str
    image_sha256: str
    width: int
    height: int
    voice_group: str
    created_at: float
    expires_at: float
    state: str = "ready"
    owner_sessionid: Optional[str] = None


def _detected_format(payload: bytes) -> Optional[str]:
    if payload.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "webp"
    return None


class CameraCaptureManager:
    """Owns capture metadata, trusted temp paths, claiming, TTL and deletion."""

    def __init__(
        self,
        root: str | Path | None = None,
        validator: Optional[Callable[[str], object]] = None,
        reference_releaser: Optional[Callable[[str], None]] = None,
        ttl_seconds: int = CAPTURE_TTL_SECONDS,
        max_captures: int = 16,
        clock: Callable[[], float] = time.time,
    ):
        configured_root = (
            root
            or os.environ.get("LIVETALKING_CAMERA_CAPTURE_ROOT")
            or Path(tempfile.gettempdir()) / "livetalking-avatar-captures"
        )
        self.root = Path(configured_root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.validator = validator
        self.reference_releaser = reference_releaser
        self.ttl_seconds = int(ttl_seconds)
        self.max_captures = int(max_captures)
        self.clock = clock
        self._records: dict[str, CameraCaptureRecord] = {}
        self._upload_attempts: dict[str, list[float]] = {}
        self._lock = RLock()

    def create(self, payload: bytes, voice_group: str) -> CameraCaptureRecord:
        self.cleanup_expired()
        with self._lock:
            if len(self._records) >= self.max_captures:
                raise CameraCaptureError("capture_capacity_reached")
        if voice_group not in VOICE_GROUPS:
            raise CameraCaptureError("invalid_voice_group")
        if not payload:
            raise CameraCaptureError("image_decode_failed")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise CameraCaptureError("camera_image_too_large")
        if _detected_format(payload) is None:
            raise CameraCaptureError("unsupported_image_format")

        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise CameraCaptureError("image_decode_failed")
        height, width = frame.shape[:2]
        if width * height > MAX_DECODED_PIXELS:
            raise CameraCaptureError("camera_image_too_large")
        if min(width, height) < MIN_SHORT_EDGE:
            raise CameraCaptureError("face_too_small")
        aspect = width / height
        if aspect < 0.4 or aspect > 2.5:
            raise CameraCaptureError("unsupported_image_format")

        longest = max(width, height)
        if longest > MAX_LONG_EDGE:
            scale = MAX_LONG_EDGE / longest
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = frame.shape[:2]

        capture_id = uuid.uuid4().hex
        capture_dir = self.root / capture_id
        image_path = capture_dir / "reference.jpg"
        capture_dir.mkdir(mode=0o700)
        try:
            if not cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise CameraCaptureError("image_decode_failed")
            normalized_payload = image_path.read_bytes()
            if self.validator is not None:
                try:
                    prepared = self.validator(str(image_path))
                except CameraCaptureError:
                    raise
                except Exception as exc:
                    error_code = getattr(exc, "code", "no_face_detected")
                    if error_code not in {
                        "no_face_detected",
                        "multiple_faces_detected",
                        "face_too_small",
                        "face_confidence_too_low",
                    }:
                        error_code = "no_face_detected"
                    raise CameraCaptureError(error_code) from exc
                face_frame = getattr(prepared, "face_frame_bgr", None)
                if face_frame is None or np.asarray(face_frame).size == 0:
                    raise CameraCaptureError("no_face_detected")

            now = self.clock()
            record = CameraCaptureRecord(
                capture_id=capture_id,
                image_path=str(image_path),
                image_sha256=hashlib.sha256(normalized_payload).hexdigest(),
                width=width,
                height=height,
                voice_group=voice_group,
                created_at=now,
                expires_at=now + self.ttl_seconds,
            )
            with self._lock:
                if len(self._records) >= self.max_captures:
                    raise CameraCaptureError("capture_capacity_reached")
                self._records[capture_id] = record
            return record
        except Exception:
            self._release_reference(str(image_path))
            shutil.rmtree(capture_dir, ignore_errors=True)
            raise

    def check_rate_limit(
        self,
        client_key: str,
        limit: int = 6,
        window_seconds: int = 60,
    ) -> None:
        now = self.clock()
        cutoff = now - window_seconds
        with self._lock:
            attempts = [
                attempt
                for attempt in self._upload_attempts.get(client_key, [])
                if attempt > cutoff
            ]
            if len(attempts) >= limit:
                raise CameraCaptureError("capture_rate_limited")
            attempts.append(now)
            self._upload_attempts[client_key] = attempts

    def claim(self, capture_id: str, sessionid: str) -> CameraCaptureRecord:
        now = self.clock()
        with self._lock:
            record = self._records.get(capture_id)
            if record is None:
                raise CameraCaptureError("capture_not_found")
            if record.expires_at <= now:
                record.state = "expired"
                raise CameraCaptureError("capture_expired")
            if record.state != "ready":
                raise CameraCaptureError("capture_already_claimed")
            record.state = "claimed"
            record.owner_sessionid = sessionid
            return record

    def release_claim(self, capture_id: str, sessionid: str) -> None:
        with self._lock:
            record = self._records.get(capture_id)
            if (
                record is not None
                and record.state == "claimed"
                and record.owner_sessionid == sessionid
                and record.expires_at > self.clock()
            ):
                record.state = "ready"
                record.owner_sessionid = None

    def destroy(
        self,
        capture_id: str,
        owner_sessionid: Optional[str] = None,
    ) -> bool:
        with self._lock:
            record = self._records.get(capture_id)
            if record is None:
                return False
            if (
                owner_sessionid is not None
                and record.owner_sessionid not in (None, owner_sessionid)
            ):
                raise CameraCaptureError("capture_owner_mismatch")
            self._records.pop(capture_id, None)
        self._release_reference(record.image_path)
        shutil.rmtree(Path(record.image_path).parent, ignore_errors=True)
        return True

    def cancel_ready(self, capture_id: str) -> bool:
        """Cancel an upload that has not been bound to a WebRTC session."""
        with self._lock:
            record = self._records.get(capture_id)
            if record is None:
                return False
            if record.state != "ready":
                raise CameraCaptureError("capture_already_claimed")
            self._records.pop(capture_id, None)
        self._release_reference(record.image_path)
        shutil.rmtree(Path(record.image_path).parent, ignore_errors=True)
        return True

    def cleanup_expired(self) -> int:
        now = self.clock()
        with self._lock:
            expired_ids = [
                capture_id
                for capture_id, record in self._records.items()
                if record.expires_at <= now and record.state != "claimed"
            ]
            rate_cutoff = now - 60
            self._upload_attempts = {
                key: [attempt for attempt in attempts if attempt > rate_cutoff]
                for key, attempts in self._upload_attempts.items()
                if any(attempt > rate_cutoff for attempt in attempts)
            }
        for capture_id in expired_ids:
            self.destroy(capture_id)
        return len(expired_ids)

    def get(self, capture_id: str) -> Optional[CameraCaptureRecord]:
        with self._lock:
            return self._records.get(capture_id)

    def close(self) -> None:
        with self._lock:
            capture_ids = list(self._records)
        for capture_id in capture_ids:
            self.destroy(capture_id)

    def _release_reference(self, image_path: str) -> None:
        if self.reference_releaser is not None:
            self.reference_releaser(image_path)
