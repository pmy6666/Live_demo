###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
from typing import Dict, Optional
import queue

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager

class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()
        self.session_pcs: dict[str, RTCPeerConnection] = {}

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        #sessionid = _rand_session_id()

        # 通过 SessionManager 构建
        try:
            sessionid = await session_manager.create_session(params)
        except RuntimeError as exc:
            logger.warning("session creation rejected: %s", exc)
            status = 429 if "maximum session limit" in str(exc) else 400
            return web.Response(
                status=status,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": str(exc)}),
            )
        except Exception as exc:
            logger.warning("session creation failed: %s", exc)
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": str(exc)}),
            )
        logger.info('offer sessionid=%s', sessionid)
        avatar_session = session_manager.get_session(sessionid)

        # 创建 PeerConnection
        ice_server = RTCIceServer(urls='stun:stun.freeswitch.org:3478')
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[ice_server])
        )
        self.pcs.add(pc)
        self.session_pcs[sessionid] = pc

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await pc.close()
                self.pcs.discard(pc)
                self.session_pcs.pop(sessionid, None)
                session_manager.remove_session(sessionid)

        # 添加发送轨道
        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        # 设置编解码器偏好
        capabilities = RTCRtpSender.getCapabilities("video")
        preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        transceiver = pc.getTransceivers()[1]
        transceiver.setCodecPreferences(preferences)

        try:
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as exc:
            logger.exception("WebRTC negotiation failed: sessionid=%s", sessionid)
            await pc.close()
            self.pcs.discard(pc)
            self.session_pcs.pop(sessionid, None)
            session_manager.remove_session(sessionid)
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"webrtc_negotiation_failed: {exc}"}),
            )

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
                "avatar": getattr(avatar_session.opt, "avatar_id", ""),
            }),
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
        self.session_pcs.clear()
        for item in session_manager.list_sessions():
            session_manager.remove_session(item["sessionid"])

    async def close_session(self, sessionid: str) -> bool:
        pc = self.session_pcs.pop(sessionid, None)
        if pc is None:
            return False
        await pc.close()
        self.pcs.discard(pc)
        session_manager.remove_session(sessionid)
        return True
