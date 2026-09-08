import asyncio
import json
import os
from pathlib import Path

from aiohttp import web

from choice.avatar_profiles import avatar_profiles, get_avatar_profile
from choice.orchestrator import ChoiceConflict
from server.camera_capture_manager import (
    MAX_UPLOAD_BYTES,
    CameraCaptureError,
)
from server.session_manager import session_manager
from utils.logger import logger


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AVATAR_ROOT = PROJECT_ROOT / "data" / "avatars"
DEFAULT_ASSET_AVATAR_ROOT = PROJECT_ROOT / "assets" / "avatars"

ASSET_AVATAR_DISPLAY = {
    "avatar1": {
        "name": "角色 1",
        "description": "董卿音色数字人",
    },
    "avatar2": {
        "name": "角色 2",
        "description": "女性音色数字人",
    },
    "avatar3": {
        "name": "角色 3",
        "description": "女性音色数字人",
    },
    "avatar4": {
        "name": "角色 4",
        "description": "撒贝宁音色数字人",
    },
    "avatar5": {
        "name": "角色 5",
        "description": "撒贝宁音色数字人",
    },
    "avatar6": {
        "name": "角色 6",
        "description": "撒贝宁增强音色数字人",
    },
    "avatar7": {
        "name": "角色 7",
        "description": "静音微呼吸数字人",
    },
}


def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(content_type="application/json", text=json.dumps(body))


def json_error(msg: str, code: int = -1, status: int = 200, data=None):
    body = {"code": code, "msg": str(msg)}
    if data is not None:
        body["data"] = data
    return web.Response(
        status=status,
        content_type="application/json",
        text=json.dumps(body),
    )


def parse_json_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError("boolean field must be true or false")


def get_session(request, sessionid: str):
    return session_manager.get_session(sessionid)


def resolve_choice_graph_id(avatar_session, params: dict) -> str:
    """Keep the registered avatar graph authoritative over stale browser assets."""
    configured = getattr(avatar_session.opt, "choice_graph_id", "")
    if configured:
        return configured
    requested = params.get("graph_id") or params.get("tree_id")
    if requested:
        return requested
    return getattr(avatar_session.opt, "choice_tree_id", "daily_chat")


def resolve_avatar_root() -> Path:
    env_root = os.environ.get("LIVETALKING_AVATAR_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(DEFAULT_AVATAR_ROOT)
    candidates.append(Path("data/avatars"))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_AVATAR_ROOT


def natural_sort_key(value: str):
    parts = []
    current = ""
    current_is_digit = None
    for ch in value:
        is_digit = ch.isdigit()
        if current_is_digit is None or current_is_digit == is_digit:
            current += ch
        else:
            parts.append(int(current) if current_is_digit else current.lower())
            current = ch
        current_is_digit = is_digit
    if current:
        parts.append(int(current) if current_is_digit else current.lower())
    return parts


def build_preview_url(image_path: Path, avatar_root: Path) -> str:
    relative_path = image_path.relative_to(avatar_root).as_posix()
    return f"/avatar-data/{relative_path}"


def build_asset_preview_url(image_path: Path, avatar_root: Path) -> str:
    relative_path = image_path.relative_to(avatar_root).as_posix()
    return f"/avatar-assets/{relative_path}"


def pick_avatar_preview(avatar_dir: Path, avatar_root: Path):
    search_dirs = [
        avatar_dir / "preview",
        avatar_dir / "full_imgs",
        avatar_dir / "face_imgs",
        avatar_dir,
    ]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        image_files = sorted(
            [
                item
                for item in search_dir.iterdir()
                if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ],
            key=lambda item: natural_sort_key(item.name),
        )
        if image_files:
            return build_preview_url(image_files[0], avatar_root)
    return None


async def list_avatars(request):
    try:
        runtime_model = request.app.get("runtime_model", "")
        configured = avatar_profiles.list(model=runtime_model, enabled_only=True)
        if configured:
            avatars = []
            if runtime_model == "avatarforcing":
                avatars.append(
                    {
                        "id": "avatarforcing_camera",
                        "name": "自拍或上传图片",
                        "description": "拍摄或上传一张正面人像，使用离线男声或女声进行选项对话",
                        "image": None,
                        "available": True,
                        "camera_capture": True,
                        "graph_id": "daily_chat",
                    }
                )
            for profile in configured:
                preview_path = Path(profile.get("preview_image_path", ""))
                available = preview_path.is_file() and Path(
                    profile.get("reference_image_path", "")
                ).is_file() and Path(profile.get("voice", {}).get("ref_file_path", "")).is_file()
                avatars.append(
                    {
                        "id": profile["avatar_id"],
                        "name": profile.get("display_name", profile["avatar_id"]),
                        "description": profile.get("description", "数字人角色"),
                        "image": f"/api/avatar-preview/{profile['avatar_id']}" if preview_path.is_file() else None,
                        "available": available,
                        "reason": None if available else "角色资源尚未就绪",
                        "graph_id": profile.get("choice", {}).get("graph_id"),
                    }
                )
            return json_ok(data={"avatars": avatars, "source": "avatar_profiles"})

        asset_root = DEFAULT_ASSET_AVATAR_ROOT
        if asset_root.exists():
            asset_images = sorted(
                [
                    item
                    for item in asset_root.iterdir()
                    if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTENSIONS
                ],
                key=lambda item: natural_sort_key(item.stem),
            )
            if asset_images:
                avatars = []
                for index, image_path in enumerate(asset_images, start=1):
                    display = ASSET_AVATAR_DISPLAY.get(image_path.stem, {})
                    avatars.append(
                        {
                            "id": image_path.stem,
                            "name": display.get("name", image_path.stem),
                            "description": display.get("description", f"EchoMimicV3 数字人角色 {index}"),
                            "image": build_asset_preview_url(image_path, asset_root),
                        }
                    )
                return json_ok(data={"avatars": avatars, "avatar_root": str(asset_root.resolve())})

        avatar_root = resolve_avatar_root()
        if not avatar_root.exists():
            return json_ok(data={"avatars": [], "avatar_root": str(avatar_root)})

        avatar_dirs = sorted(
            [item for item in avatar_root.iterdir() if item.is_dir()],
            key=lambda item: natural_sort_key(item.name),
        )

        avatars = []
        for index, avatar_dir in enumerate(avatar_dirs, start=1):
            avatars.append(
                {
                    "id": avatar_dir.name,
                    "name": avatar_dir.name,
                    "description": f"数字人角色 {index}",
                    "image": pick_avatar_preview(avatar_dir, avatar_root),
                }
            )

        return json_ok(data={"avatars": avatars, "avatar_root": str(avatar_root)})
    except Exception as exc:
        logger.exception("list_avatars exception:")
        return json_error(str(exc))


async def avatar_preview(request):
    avatar_id = request.match_info.get("avatar_id", "")
    profile = get_avatar_profile(avatar_id, enabled_only=True)
    if not profile:
        raise web.HTTPNotFound()
    path = Path(profile.get("preview_image_path", ""))
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def list_sessions(request):
    try:
        sessions = session_manager.list_sessions()
        return json_ok(
            data={
                "sessions": sessions,
                "active_sessionid": session_manager.latest_ready_sessionid(),
            }
        )
    except Exception as exc:
        logger.exception("list_sessions exception:")
        return json_error(str(exc))


async def human(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        logger.info(
            "human request type=%s sessionid=%s text_len=%d interrupt=%s",
            params.get("type"),
            sessionid,
            len(params.get("text", "")),
            parse_json_bool(params.get("interrupt"), False),
        )
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.warning("human request rejected: session not found, sessionid=%s", sessionid)
            return json_error("session not found")
        if getattr(avatar_session.opt, "conversation_mode", "") == "offline_choice_only":
            return json_error("offline_choice_only")

        if parse_json_bool(params.get("interrupt"), False):
            avatar_session.flush_talk(reason="human_interrupt")

        datainfo = {}
        if params.get("tts"):
            datainfo["tts"] = params.get("tts")

        if params["type"] == "echo":
            avatar_session.put_msg_txt(params["text"], datainfo)
        elif params["type"] == "chat":
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params["text"], avatar_session, datainfo
                )

        return json_ok()
    except Exception as exc:
        logger.exception("human route exception:")
        return json_error(str(exc))


async def interrupt_talk(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        old_playback_id = avatar_session.current_playback_token()
        playback_id = avatar_session.flush_talk(reason=params.get("reason", "user_stop"))
        return json_ok(data={"playback_id": playback_id, "interrupted_playback_id": old_playback_id})
    except Exception as exc:
        logger.exception("interrupt_talk exception:")
        return json_error(str(exc))


async def humanaudio(request):
    try:
        form = await request.post()
        sessionid = str(form.get("sessionid", ""))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if getattr(avatar_session.opt, "conversation_mode", "") == "offline_choice_only":
            return json_error("offline_choice_only")
        avatar_session.put_audio_file(filebytes, {})
        return json_ok()
    except Exception as exc:
        logger.exception("humanaudio exception:")
        return json_error(str(exc))


async def set_audiotype(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params["audiotype"])
        return json_ok()
    except Exception as exc:
        logger.exception("set_audiotype exception:")
        return json_error(str(exc))


async def record(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params["type"] == "start_record":
            avatar_session.start_recording()
        elif params["type"] == "end_record":
            avatar_session.stop_recording()
        return json_ok()
    except Exception as exc:
        logger.exception("record exception:")
        return json_error(str(exc))


async def is_speaking(request):
    params = await request.json()
    sessionid = params.get("sessionid", "")
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


async def choice_init(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        tree_id = resolve_choice_graph_id(avatar_session, params)
        requested_tree_id = params.get("graph_id") or params.get("tree_id")
        if requested_tree_id and requested_tree_id != tree_id:
            logger.info(
                "choice init ignored stale client graph sessionid=%s requested=%s configured=%s",
                sessionid,
                requested_tree_id,
                tree_id,
            )
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        payload = orchestrator.init_session(avatar_session, tree_id)
        return json_ok(data=payload)
    except Exception as exc:
        logger.exception("choice_init exception:")
        return json_error(str(exc))


async def choice_select(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        choice_id = params.get("choice_id", "")
        logger.info("choice select request sessionid=%s choice_id=%s", sessionid, choice_id)
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.warning("choice select rejected: session not found, sessionid=%s", sessionid)
            return json_error("session not found")
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        payload = orchestrator.select_choice(
            avatar_session,
            choice_id=choice_id,
            interrupt=parse_json_bool(params.get("interrupt"), True),
            client_seq=(int(params["client_seq"]) if params.get("client_seq") is not None else None),
            expected_state_version=(
                int(params["expected_state_version"])
                if params.get("expected_state_version") is not None
                else None
            ),
        )
        return json_ok(data=payload)
    except ChoiceConflict as exc:
        logger.info("choice_select conflict: %s", exc)
        return json_error(str(exc), code=409, status=409, data=exc.current_state)
    except Exception as exc:
        logger.exception("choice_select exception:")
        return json_error(str(exc))


async def choice_state(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        payload = orchestrator.get_state(avatar_session)
        return json_ok(data=payload)
    except Exception as exc:
        logger.exception("choice_state exception:")
        return json_error(str(exc))


async def choice_reset(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        avatar_session.flush_talk(reason="choice_reset")
        payload = orchestrator.reset_session(avatar_session)
        return json_ok(data=payload)
    except Exception as exc:
        logger.exception("choice_reset exception:")
        return json_error(str(exc))


async def create_avatar_capture(request):
    manager = request.app.get("camera_capture_manager")
    if manager is None:
        return json_error("camera_capture_unavailable", status=503)
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES + 64 * 1024:
        return json_error("camera_image_too_large", status=413)
    if not request.content_type.startswith("multipart/"):
        return json_error("unsupported_image_format", status=415)

    image = None
    voice_group = ""
    try:
        manager.check_rate_limit(request.remote or "unknown")
        reader = await request.multipart()
        async for field in reader:
            if field.name == "voice_group":
                voice_group = (await field.text()).strip()
            elif field.name == "image":
                chunks = bytearray()
                while True:
                    chunk = await field.read_chunk(size=64 * 1024)
                    if not chunk:
                        break
                    chunks.extend(chunk)
                    if len(chunks) > MAX_UPLOAD_BYTES:
                        raise CameraCaptureError("camera_image_too_large")
                image = bytes(chunks)
        if image is None:
            raise CameraCaptureError("image_decode_failed")
        semaphore = request.app["camera_capture_semaphore"]
        async with semaphore:
            record = await asyncio.get_running_loop().run_in_executor(
                None,
                manager.create,
                image,
                voice_group,
            )
        return json_ok(
            data={
                "capture_id": record.capture_id,
                "width": record.width,
                "height": record.height,
                "face_ready": True,
                "expires_in": max(0, round(record.expires_at - manager.clock())),
            }
        )
    except CameraCaptureError as exc:
        return json_error(exc.code, status=400)
    except Exception:
        logger.exception("create_avatar_capture exception:")
        return json_error("capture_validation_failed", status=400)


async def cancel_avatar_capture(request):
    manager = request.app.get("camera_capture_manager")
    if manager is None:
        return json_error("camera_capture_unavailable", status=503)
    try:
        params = await request.json()
        capture_id = str(params.get("capture_id", ""))
        if not capture_id:
            return json_error("capture_not_found", status=404)
        removed = manager.cancel_ready(capture_id)
        if not removed:
            return json_error("capture_not_found", status=404)
        return json_ok()
    except CameraCaptureError as exc:
        return json_error(exc.code, status=409)
    except Exception:
        logger.exception("cancel_avatar_capture exception:")
        return json_error("capture_cancel_failed", status=400)


async def close_session(request):
    try:
        params = await request.json()
        sessionid = str(params.get("sessionid", ""))
        rtc_manager = request.app.get("rtc_manager")
        if not sessionid or rtc_manager is None:
            return json_error("session not found", status=404)
        closed = await rtc_manager.close_session(sessionid)
        if not closed:
            return json_error("session not found", status=404)
        return json_ok()
    except Exception:
        logger.exception("close_session exception:")
        return json_error("session_close_failed", status=400)


async def camera_capture_cleanup_context(app):
    manager = app.get("camera_capture_manager")
    if manager is None:
        yield
        return

    async def cleanup_loop():
        while True:
            await asyncio.sleep(30)
            removed = manager.cleanup_expired()
            if removed:
                logger.info("expired camera captures removed: count=%d", removed)

    task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        manager.close()


def setup_routes(app):
    avatar_root = resolve_avatar_root()
    asset_avatar_root = DEFAULT_ASSET_AVATAR_ROOT
    app.cleanup_ctx.append(camera_capture_cleanup_context)
    app["camera_capture_semaphore"] = asyncio.Semaphore(2)

    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_post("/choice/init", choice_init)
    app.router.add_post("/choice/select", choice_select)
    app.router.add_post("/choice/state", choice_state)
    app.router.add_post("/choice/reset", choice_reset)
    app.router.add_get("/api/avatars", list_avatars)
    app.router.add_get("/api/avatar-preview/{avatar_id}", avatar_preview)
    app.router.add_get("/api/sessions", list_sessions)
    app.router.add_post("/api/avatar-captures", create_avatar_capture)
    app.router.add_post("/api/avatar-captures/cancel", cancel_avatar_capture)
    app.router.add_post("/api/session/close", close_session)

    if avatar_root.exists():
        app.router.add_static("/avatar-data/", path=str(avatar_root))
    if asset_avatar_root.exists():
        app.router.add_static("/avatar-assets/", path=str(asset_avatar_root))

    app.router.add_static("/", path="web")
