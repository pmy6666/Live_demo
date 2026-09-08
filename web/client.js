var pc = null;
var currentAvatar = null;
var avatarCatalog = [];
var cameraState = {
    stream: null,
    photoBlob: null,
    photoFilename: "camera.jpg",
    photoUrl: null,
    captureId: null,
    busy: false
};
var videoViewportState = {
    baseVideoWidth: 0,
    baseVideoHeight: 0,
    defaultViewportHeight: 0,
    minWidth: 280,
    minHeight: 220,
    maxHeight: 720
};
var choiceState = {
    initialized: false,
    treeId: "daily_chat",
    current: null,
    path: [],
    stateVersion: 0,
    clientSeq: 0,
    playbackId: 0
};
var ssvepState = {
    enabled: false,
    originalColor: false,
    rafId: null,
    frameCnt: 0,
    actualFps: 60,
    lutLen: 1000,
    defaultFrequencies: [12.8, 11.2, 8.8],
    defaultPhases: [0, 0, 0],
    targets: []
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function(ch) {
        return {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "\"": "&quot;",
            "'": "&#39;"
        }[ch];
    });
}

function buildAvatarPlaceholder(avatar) {
    var title = escapeHtml(avatar.name || avatar.id || "Avatar");
    return "data:image/svg+xml;charset=UTF-8," + encodeURIComponent(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 600'>" +
        "<defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>" +
        "<stop stop-color='#dce8ff' offset='0%'/>" +
        "<stop stop-color='#f7fbff' offset='100%'/>" +
        "</linearGradient></defs>" +
        "<rect width='480' height='600' fill='url(#g)'/>" +
        "<circle cx='240' cy='190' r='58' fill='#1149d8' opacity='0.75'/>" +
        "<rect x='150' y='266' width='180' height='132' rx='58' fill='#1149d8' opacity='0.72'/>" +
        "<text x='240' y='470' font-size='34' text-anchor='middle' fill='#122033' font-family='Segoe UI, Arial, sans-serif' font-weight='700'>" + title + "</text>" +
        "<text x='240' y='512' font-size='20' text-anchor='middle' fill='#66758a' font-family='Segoe UI, Arial, sans-serif'>LiveTalking</text>" +
        "</svg>"
    );
}

function normalizeAvatarCatalog(avatars) {
    return avatars.map(function(avatar, index) {
        var normalized = {
            id: avatar.id,
            name: avatar.name || avatar.id,
            description: avatar.description || ("数字人角色 " + (index + 1)),
            image: avatar.image || null,
            available: avatar.available !== false,
            reason: avatar.reason || ""
        };
        normalized.cameraCapture = avatar.camera_capture === true;
        normalized.placeholder = buildAvatarPlaceholder(normalized);
        normalized.badge = "角色 " + (index + 1);
        return normalized;
    });
}

function getAvatarById(avatarId) {
    return avatarCatalog.find(function(item) {
        return item.id === avatarId;
    }) || null;
}

function ensureCurrentAvatar() {
    if (!currentAvatar && avatarCatalog.length > 0) {
        currentAvatar = avatarCatalog[0];
    }
}

function isCameraAvatar(avatar) {
    return Boolean(avatar && avatar.cameraCapture);
}

function setCameraStatus(message, isError) {
    $("#camera-status")
        .text(message)
        .toggleClass("error", Boolean(isError));
}

function stopCameraTracks() {
    if (cameraState.stream) {
        cameraState.stream.getTracks().forEach(function(track) {
            track.stop();
        });
        cameraState.stream = null;
    }
    var preview = document.getElementById("camera-preview");
    if (preview) {
        preview.srcObject = null;
    }
}

function resetCameraPhoto() {
    cameraState.photoBlob = null;
    cameraState.photoFilename = "camera.jpg";
    if (cameraState.photoUrl) {
        URL.revokeObjectURL(cameraState.photoUrl);
        cameraState.photoUrl = null;
    }
    $("#camera-photo-preview").prop("hidden", true).attr("src", "");
    $("#camera-preview, #camera-guide").prop("hidden", false);
    $("#camera-retake, #camera-confirm").prop("hidden", true);
    $("#camera-take-photo").prop("hidden", false).prop("disabled", !cameraState.stream);
}

function selectUploadedAvatarImage(event) {
    var file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file || cameraState.busy) {
        return;
    }
    var supportedTypes = {
        "image/jpeg": "uploaded.jpg",
        "image/png": "uploaded.png",
        "image/webp": "uploaded.webp"
    };
    var uploadFilename = supportedTypes[file.type];
    if (!uploadFilename) {
        setCameraStatus("请选择 JPEG、PNG 或 WebP 图片。", true);
        return;
    }
    if (file.size > 8 * 1024 * 1024) {
        setCameraStatus("图片不能超过 8 MiB，请压缩后重新选择。", true);
        return;
    }

    cancelUnclaimedCapture();
    stopCameraTracks();
    resetCameraPhoto();
    cameraState.photoBlob = file;
    cameraState.photoFilename = uploadFilename;
    cameraState.photoUrl = URL.createObjectURL(file);
    $("#camera-capture-panel").addClass("active");
    $("#enter-chat-btn").prop("disabled", true);
    $("#camera-photo-preview")
        .attr("src", cameraState.photoUrl)
        .prop("hidden", false);
    $("#camera-preview, #camera-guide, #camera-take-photo, #camera-retake")
        .prop("hidden", true);
    $("#camera-confirm").prop("hidden", false);
    setCameraStatus(
        "本地图片已选择。图片不会镜像，播放时使用不贴回模式。",
        false
    );
}

function cancelUnclaimedCapture() {
    var captureId = cameraState.captureId;
    cameraState.captureId = null;
    if (!captureId || ensureSessionReady()) {
        return Promise.resolve();
    }
    return postJson("/api/avatar-captures/cancel", {
        capture_id: captureId
    }).catch(function() {
        // TTL cleanup remains authoritative if the browser disappears mid-request.
    });
}

function cameraErrorMessage(code) {
    var messages = {
        camera_image_too_large: "图片过大，请压缩、重拍或重新选择。",
        unsupported_image_format: "图片格式不受支持，请使用 JPEG、PNG 或 WebP。",
        image_decode_failed: "图片无法读取，请重拍或重新选择。",
        no_face_detected: "未检测到清晰正脸，请调整位置、光线或重新选择图片。",
        multiple_faces_detected: "画面中检测到多张人脸，请只保留一人。",
        face_too_small: "人脸或图片尺寸太小，请靠近镜头或重新选择图片。",
        face_confidence_too_low: "人脸清晰度不足，请调整光线或重新选择图片。",
        invalid_voice_group: "请选择男声或女声。",
        capture_rate_limited: "拍照上传过于频繁，请一分钟后再试。",
        capture_capacity_reached: "当前正在处理的人像较多，请稍后再试。"
    };
    return messages[code] || code;
}

function openCameraCapture() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraStatus("当前浏览器或访问环境不支持摄像头。请使用 HTTPS 或 localhost。", true);
        $("#camera-capture-panel").addClass("active");
        return;
    }
    cancelUnclaimedCapture();
    stopCameraTracks();
    resetCameraPhoto();
    $("#camera-capture-panel").addClass("active");
    $("#enter-chat-btn").prop("disabled", true);
    setCameraStatus("正在请求摄像头权限…", false);
    navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 }
        },
        audio: false
    }).then(function(stream) {
        cameraState.stream = stream;
        document.getElementById("camera-preview").srcObject = stream;
        $("#camera-take-photo").prop("disabled", false);
        setCameraStatus("摄像头已就绪。自拍会镜像处理，本地上传图片保持原方向。", false);
    }).catch(function(error) {
        console.error(error);
        setCameraStatus("无法使用摄像头：" + (error.message || error.name), true);
    });
}

function captureCameraPhoto() {
    if (!cameraState.stream || cameraState.busy) {
        return;
    }
    cameraState.busy = true;
    $("#camera-take-photo").prop("disabled", true);
    var countdown = $("#camera-countdown").addClass("active");
    var seconds = 3;
    countdown.text(seconds);
    var timer = window.setInterval(function() {
        seconds -= 1;
        if (seconds > 0) {
            countdown.text(seconds);
            return;
        }
        window.clearInterval(timer);
        countdown.removeClass("active").text("");
        var video = document.getElementById("camera-preview");
        var canvas = document.getElementById("camera-canvas");
        var sourceWidth = video.videoWidth;
        var sourceHeight = video.videoHeight;
        if (!sourceWidth || !sourceHeight) {
            cameraState.busy = false;
            $("#camera-take-photo").prop("disabled", false);
            setCameraStatus("摄像头画面尚未就绪，请稍后重试。", true);
            return;
        }
        var scale = Math.min(1, 1920 / Math.max(sourceWidth, sourceHeight));
        canvas.width = Math.round(sourceWidth * scale);
        canvas.height = Math.round(sourceHeight * scale);
        var context = canvas.getContext("2d");
        context.save();
        context.translate(canvas.width, 0);
        context.scale(-1, 1);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        context.restore();
        canvas.toBlob(function(blob) {
            cameraState.busy = false;
            if (!blob) {
                setCameraStatus("拍照失败，请重试。", true);
                $("#camera-take-photo").prop("disabled", false);
                return;
            }
            cameraState.photoBlob = blob;
            cameraState.photoFilename = "camera.jpg";
            cameraState.photoUrl = URL.createObjectURL(blob);
            $("#camera-photo-preview")
                .attr("src", cameraState.photoUrl)
                .prop("hidden", false);
            $("#camera-preview, #camera-guide, #camera-take-photo").prop("hidden", true);
            $("#camera-retake, #camera-confirm").prop("hidden", false);
            setCameraStatus("照片已拍摄。确认构图和音色后即可上传。", false);
        }, "image/jpeg", 0.92);
    }, 1000);
}

function uploadCameraCapture() {
    if (!cameraState.photoBlob || cameraState.busy) {
        return;
    }
    cameraState.busy = true;
    $(
        "#camera-confirm, #camera-retake, #camera-cancel, " +
        "#camera-upload-image, #camera-take-photo"
    ).prop("disabled", true);
    setCameraStatus("正在上传并检查人脸…", false);
    var voiceGroup = $('input[name="camera-voice-group"]:checked').val();
    var form = new FormData();
    form.append("image", cameraState.photoBlob, cameraState.photoFilename);
    form.append("voice_group", voiceGroup);
    fetch("/api/avatar-captures", {
        method: "POST",
        body: form
    }).then(function(response) {
        return response.text().then(function(text) {
            var payload = {};
            try {
                payload = JSON.parse(text);
            } catch (error) {
                throw new Error("capture_validation_failed");
            }
            if (!response.ok || payload.code !== 0) {
                throw new Error(payload.msg || "capture_validation_failed");
            }
            return payload.data;
        });
    }).then(function(data) {
        cameraState.captureId = data.capture_id;
        var cameraAvatar = avatarCatalog.find(isCameraAvatar);
        cameraAvatar.image = cameraState.photoUrl;
        setCurrentAvatar(cameraAvatar.id);
        stopCameraTracks();
        $("#camera-capture-panel").removeClass("active");
        setCameraStatus("人脸检查通过，正在建立离线对话…", false);
        switchPage("conversation");
        start();
    }).catch(function(error) {
        console.error(error);
        setCameraStatus(cameraErrorMessage(error.message), true);
    }).finally(function() {
        cameraState.busy = false;
        $("#camera-confirm, #camera-retake, #camera-cancel, #camera-upload-image")
            .prop("disabled", false);
        $("#camera-take-photo").prop("disabled", !cameraState.stream);
    });
}

function updateConnectionStatus(status) {
    var statusIndicator = $("#connection-status");
    var statusText = $("#status-text");

    statusIndicator.removeClass("status-connected status-disconnected status-connecting");

    switch (status) {
        case "connected":
            statusIndicator.addClass("status-connected");
            statusText.text("已连接");
            break;
        case "connecting":
            statusIndicator.addClass("status-connecting");
            statusText.text("连接中...");
            break;
        default:
            statusIndicator.addClass("status-disconnected");
            statusText.text("未连接");
            break;
    }
}

function addChatMessage(message, type) {
    var messageType = type || "user";
    var messageClass = messageType === "user" ? "user-message" : "system-message";
    var sender = messageType === "user" ? "你" : "系统";
    var messageElement = $(
        '<div class="asr-text ' + messageClass + '">' +
            escapeHtml(sender) + "：" + escapeHtml(message) +
        "</div>"
    );

    $("#chat-messages").append(messageElement);
    var container = document.getElementById("chat-messages");
    container.scrollTop = container.scrollHeight;
}

function ensureSessionReady() {
    var sessionid = String(document.getElementById("sessionid").value || "0");
    return sessionid && sessionid !== "0";
}

function getViewportElements() {
    return {
        shell: document.getElementById("video-viewport"),
        wrapper: document.getElementById("video-viewport-wrap"),
        video: document.getElementById("video")
    };
}

function updateViewportControls() {
    var elements = getViewportElements();
    if (!elements.shell || !elements.wrapper) {
        return;
    }
    var width = elements.shell.offsetWidth;
    var height = elements.shell.offsetHeight;
    var availableWidth = Math.max(1, elements.wrapper.clientWidth);
    var widthPercent = Math.round(width / availableWidth * 100);
    $("#video-viewport-width-slider").val(Math.max(30, Math.min(100, widthPercent)));
    $("#video-viewport-height-slider").val(Math.max(
        videoViewportState.minHeight,
        Math.min(videoViewportState.maxHeight, height)
    ));
    $("#video-viewport-width-value").text(widthPercent + "%");
    $("#video-viewport-height-value").text(Math.round(height) + "px");
    $("#video-viewport-size").text(
        Math.round(width) + " × " + Math.round(height) + "px"
    );
}

function setViewportSize(width, height) {
    var elements = getViewportElements();
    if (!elements.shell || !elements.wrapper) {
        return;
    }
    var maxWidth = Math.max(1, elements.wrapper.clientWidth);
    var minWidth = Math.min(videoViewportState.minWidth, maxWidth);
    var nextWidth = Math.max(minWidth, Math.min(maxWidth, Number(width)));
    var nextHeight = Math.max(
        videoViewportState.minHeight,
        Math.min(videoViewportState.maxHeight, Number(height))
    );
    elements.shell.style.width = Math.round(nextWidth) + "px";
    elements.shell.style.height = Math.round(nextHeight) + "px";
    elements.shell.classList.add("has-custom-size");
    updateViewportControls();
}

function captureVideoDisplaySize() {
    var elements = getViewportElements();
    if (!elements.shell || !elements.wrapper || !elements.video) {
        return;
    }
    var sourceWidth = elements.video.videoWidth;
    var sourceHeight = elements.video.videoHeight;
    if (!sourceWidth || !sourceHeight) {
        return;
    }
    var availableWidth = Math.max(1, elements.wrapper.clientWidth - 12);
    var availableHeight = Math.max(
        1,
        (videoViewportState.defaultViewportHeight || elements.shell.clientHeight) - 12
    );
    var containScale = Math.min(
        availableWidth / sourceWidth,
        availableHeight / sourceHeight
    );
    videoViewportState.baseVideoWidth = sourceWidth * containScale;
    videoViewportState.baseVideoHeight = sourceHeight * containScale;
    elements.video.style.width =
        Math.round(videoViewportState.baseVideoWidth) + "px";
    elements.video.style.height =
        Math.round(videoViewportState.baseVideoHeight) + "px";
}

function initializeVideoViewport() {
    var elements = getViewportElements();
    var handle = document.getElementById("video-resize-handle");
    if (!elements.shell || !elements.wrapper || !elements.video || !handle) {
        return;
    }

    videoViewportState.defaultViewportHeight = elements.shell.offsetHeight;
    updateViewportControls();
    elements.video.addEventListener("loadedmetadata", captureVideoDisplaySize);

    function beginResize(event, resizeWidth, resizeHeight) {
        event.preventDefault();
        var startX = event.clientX;
        var startY = event.clientY;
        var startWidth = elements.shell.offsetWidth;
        var startHeight = elements.shell.offsetHeight;
        elements.shell.classList.add("is-resizing");
        elements.shell.setPointerCapture(event.pointerId);

        function resize(pointerEvent) {
            setViewportSize(
                resizeWidth
                    ? startWidth + pointerEvent.clientX - startX
                    : startWidth,
                resizeHeight
                    ? startHeight + pointerEvent.clientY - startY
                    : startHeight
            );
        }

        function finish(pointerEvent) {
            elements.shell.classList.remove("is-resizing");
            if (elements.shell.hasPointerCapture(pointerEvent.pointerId)) {
                elements.shell.releasePointerCapture(pointerEvent.pointerId);
            }
            elements.shell.removeEventListener("pointermove", resize);
            elements.shell.removeEventListener("pointerup", finish);
            elements.shell.removeEventListener("pointercancel", finish);
        }

        elements.shell.addEventListener("pointermove", resize);
        elements.shell.addEventListener("pointerup", finish);
        elements.shell.addEventListener("pointercancel", finish);
    }

    handle.addEventListener("pointerdown", function(event) {
        beginResize(event, true, true);
    });

    elements.shell.addEventListener("pointerdown", function(event) {
        if (event.target === handle || handle.contains(event.target)) {
            return;
        }
        var bounds = elements.shell.getBoundingClientRect();
        var edgeSize = 14;
        var onRightEdge = event.clientX >= bounds.right - edgeSize;
        var onBottomEdge = event.clientY >= bounds.bottom - edgeSize;
        if (onRightEdge || onBottomEdge) {
            beginResize(event, onRightEdge, onBottomEdge);
        }
    });

    handle.addEventListener("keydown", function(event) {
        var delta = event.shiftKey ? 25 : 10;
        var width = elements.shell.offsetWidth;
        var height = elements.shell.offsetHeight;
        if (event.key === "ArrowRight") {
            width += delta;
        } else if (event.key === "ArrowLeft") {
            width -= delta;
        } else if (event.key === "ArrowDown") {
            height += delta;
        } else if (event.key === "ArrowUp") {
            height -= delta;
        } else {
            return;
        }
        event.preventDefault();
        setViewportSize(width, height);
    });

    $("#video-viewport-width-slider").on("input", function() {
        var percent = Number(this.value);
        setViewportSize(
            elements.wrapper.clientWidth * percent / 100,
            elements.shell.offsetHeight
        );
    });

    $("#video-viewport-height-slider").on("input", function() {
        setViewportSize(elements.shell.offsetWidth, Number(this.value));
    });

    $("#video-viewport-reset").on("click", function() {
        elements.shell.style.width = "100%";
        elements.shell.style.height = "";
        elements.shell.classList.remove("has-custom-size");
        $("#video-size-slider").val(100).trigger("input");
        window.requestAnimationFrame(function() {
            videoViewportState.defaultViewportHeight = elements.shell.offsetHeight;
            captureVideoDisplaySize();
            updateViewportControls();
        });
    });

    window.addEventListener("resize", function() {
        window.requestAnimationFrame(updateViewportControls);
    });
}

function postJson(url, body) {
    return fetch(url, {
        body: JSON.stringify(body),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.text().then(function(text) {
            var payload = {};
            if (text) {
                try {
                    payload = JSON.parse(text);
                } catch (error) {
                    throw new Error(url + " 返回的不是 JSON: " + text.slice(0, 240));
                }
            }
            if (!response.ok || payload.code !== 0) {
                throw new Error(payload.msg || (url + " 请求失败: " + response.status));
            }
            return payload;
        });
    });
}

function formatFrequency(value) {
    return Number(value).toFixed(value % 1 === 0 ? 0 : 1) + "Hz";
}

function getChoiceFrequency(choice, index) {
    var ssvep = choice.ssvep || {};
    var configured = ssvep.frequency || choice.ssvep_frequency;
    var frequency = Number(configured);
    if (isFinite(frequency) && frequency > 0) {
        return frequency;
    }
    return ssvepState.defaultFrequencies[index % ssvepState.defaultFrequencies.length];
}

function getChoicePhase(choice) {
    var ssvep = choice.ssvep || {};
    var configured = ssvep.phase || choice.ssvep_phase;
    var phase = Number(configured);
    if (isFinite(phase)) {
        return phase;
    }
    return ssvepState.defaultPhases[0];
}

function buildSsvepLut(frequency, phase, actualFps, lutLen) {
    var lut = [];
    var fps = actualFps || 60;
    for (var i = 0; i < lutLen; i += 1) {
        var intensity = (
            (Math.sin(2 * Math.PI * frequency * i / fps + phase * Math.PI) + 1) / 2
        ) * 0.9 + 0.1;
        lut.push(intensity);
    }
    return lut;
}

function estimateRefreshRate(sampleFrames) {
    sampleFrames = sampleFrames || 90;
    if (!window.requestAnimationFrame) {
        return Promise.resolve(60);
    }

    return new Promise(function(resolve) {
        var times = [];

        function step(now) {
            times.push(now);
            if (times.length < sampleFrames) {
                window.requestAnimationFrame(step);
                return;
            }

            var intervals = [];
            for (var i = 1; i < times.length; i += 1) {
                intervals.push(times[i] - times[i - 1]);
            }
            intervals.sort(function(a, b) {
                return a - b;
            });

            var median = intervals[Math.floor(intervals.length / 2)] || 16.6667;
            var fps = Math.round(1000 / median);
            if (!isFinite(fps) || fps < 30 || fps > 240) {
                fps = 60;
            }
            resolve(fps);
        }

        window.requestAnimationFrame(step);
    });
}

function updateSsvepStatus() {
    var status = $("#ssvep-status");
    if (!status.length) {
        return;
    }

    if (!ssvepState.enabled) {
        status.text("关闭时按普通按钮选择；开启后选项刺激块按 LUT 正弦亮度闪烁。");
        return;
    }

    if (!ssvepState.targets.length) {
        status.text("SSVEP 已开启，等待选项加载。" + (ssvepState.originalColor ? "当前为原色显示。" : ""));
        return;
    }

    status.text("SSVEP 已开启，FPS " + Math.round(ssvepState.actualFps || 60) + "：" + ssvepState.targets.map(function(target, index) {
        return "选项 " + (index + 1) + " " + formatFrequency(target.frequency);
    }).join("，"));
}

function resetSsvepButtonVisuals() {
    $("#choice-options .choice-btn")
        .removeClass("ssvep-locked")
        .css({
            backgroundColor: "",
            borderColor: "",
            boxShadow: "",
            color: ""
        });
    $("#choice-options .ssvep-stimulus").css({
        backgroundColor: "",
        borderColor: ""
    });
}

function stopSsvepFlicker() {
    if (ssvepState.rafId !== null && window.cancelAnimationFrame) {
        window.cancelAnimationFrame(ssvepState.rafId);
        ssvepState.rafId = null;
    }
    $("#choice-options").removeClass("ssvep-enabled ssvep-original-color");
    resetSsvepButtonVisuals();
}

function startSsvepFlicker() {
    stopSsvepFlicker();
    if (!ssvepState.enabled || !ssvepState.targets.length || !window.requestAnimationFrame) {
        updateSsvepStatus();
        return;
    }

    $("#choice-options")
        .addClass("ssvep-enabled")
        .toggleClass("ssvep-original-color", ssvepState.originalColor);
    ssvepState.frameCnt = 0;
    ssvepState.actualFps = ssvepState.actualFps || 60;
    ssvepState.targets.forEach(function(target) {
        target.lut = buildSsvepLut(
            target.frequency,
            target.phase || 0,
            ssvepState.actualFps,
            ssvepState.lutLen
        );
    });

    function tick() {
        ssvepState.targets.forEach(function(target) {
            if (!target.stimulus || !target.lut) {
                return;
            }
            var intensity = target.lut[ssvepState.frameCnt % target.lut.length];
            var value = Math.round(intensity * 255);
            target.stimulus.style.backgroundColor = "rgb(" + value + "," + value + "," + value + ")";
        });
        ssvepState.frameCnt += 1;
        ssvepState.rafId = window.requestAnimationFrame(tick);
    }

    ssvepState.rafId = window.requestAnimationFrame(tick);
    updateSsvepStatus();
}

function setSsvepOriginalColor(enabled) {
    ssvepState.originalColor = Boolean(enabled);
    $("#ssvep-original-color-toggle").prop("checked", ssvepState.originalColor);
    $("#choice-options").toggleClass("ssvep-original-color", ssvepState.enabled && ssvepState.originalColor);
    updateSsvepStatus();
}

function setSsvepEnabled(enabled) {
    ssvepState.enabled = Boolean(enabled);
    $("#ssvep-toggle").prop("checked", ssvepState.enabled);
    if (ssvepState.enabled) {
        startSsvepFlicker();
        estimateRefreshRate(90).then(function(fps) {
            if (!ssvepState.enabled) {
                return;
            }
            ssvepState.actualFps = fps;
            startSsvepFlicker();
        });
    } else {
        stopSsvepFlicker();
        updateSsvepStatus();
    }
}

function selectChoiceBySsvepTarget(targetIndex) {
    var index = Number(targetIndex);
    if (!isFinite(index)) {
        return false;
    }

    if (index >= 1) {
        index -= 1;
    }

    var target = ssvepState.targets[index];
    if (!target) {
        return false;
    }

    $(target.element).addClass("ssvep-locked");
    requestChoiceSelect(target.choiceId);
    return true;
}

function renderChoiceState(payload) {
    stopSsvepFlicker();
    ssvepState.targets = [];

    if (!payload || !payload.current) {
        $("#choice-answer").text("当前还没有可用的选项对话状态。");
        $("#choice-path").text("当前路径：未初始化");
        $("#choice-options").empty();
        updateSsvepStatus();
        return;
    }

    choiceState.initialized = true;
    choiceState.current = payload.current;
    choiceState.path = payload.path || [];
    choiceState.treeId = payload.graph_id || payload.tree_id || choiceState.treeId;
    choiceState.stateVersion = Number(payload.state_version || choiceState.stateVersion || 0);
    choiceState.playbackId = Number(payload.playback_id || choiceState.playbackId || 0);

    $("#choice-answer").text(payload.current.display_text || payload.current.answer_text || "");
    $("#choice-path").text("当前路径：" + (choiceState.path.length ? choiceState.path.join(" > ") : "root"));
    var cache = payload.cache || {};
    if (cache.hit) {
        $("#choice-note").text("媒体缓存已命中（" + (cache.level || "缓存") + "），正在播放。");
    } else if (payload.current.audio_cache_hit) {
        $("#choice-note").text("当前节点音频已命中缓存，起播会更快。");
    } else {
        $("#choice-note").text("当前节点已更新；缺少视频缓存时会自动使用音频或实时生成。");
    }

    var container = $("#choice-options");
    container.empty();
    (payload.choices || payload.current.choices || []).forEach(function(choice, index) {
        var frequency = getChoiceFrequency(choice, index);
        var phase = getChoicePhase(choice);
        var button = $(
            '<button type="button" class="choice-btn ssvep-choice" data-choice-id="' + escapeHtml(choice.choice_id) + '" data-ssvep-index="' + index + '">' +
                '<span class="choice-label">' +
                    '<span class="choice-text">' +
                        '<span class="choice-number">' + (index + 1) + "</span>" +
                        escapeHtml(choice.choice_text) +
                    "</span>" +
                    '<span class="choice-ssvep-side">' +
                        '<span class="ssvep-frequency-badge">' + formatFrequency(frequency) + "</span>" +
                        '<span class="ssvep-stimulus" aria-hidden="true"></span>' +
                    "</span>" +
                "</span>" +
            "</button>"
        );
        container.append(button);
        ssvepState.targets.push({
            index: index,
            choiceId: choice.choice_id,
            frequency: frequency,
            phase: phase,
            element: button[0],
            stimulus: button.find(".ssvep-stimulus")[0]
        });
    });

    if (ssvepState.enabled) {
        startSsvepFlicker();
    } else {
        updateSsvepStatus();
    }
}

function requestChoiceInit() {
    if (!ensureSessionReady()) {
        alert("请先开始连接");
        return Promise.resolve();
    }

    return fetch("/choice/init", {
        body: JSON.stringify({
            sessionid: String(document.getElementById("sessionid").value),
            graph_id: choiceState.treeId
        }),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.json();
    }).then(function(payload) {
        if (payload.code !== 0) {
            throw new Error(payload.msg || "choice init failed");
        }
        renderChoiceState(payload.data);
        choiceState.clientSeq = 0;
    }).catch(function(error) {
        console.error(error);
        $("#choice-note").text("选项对话初始化失败：" + error.message);
    });
}

function requestChoiceReset() {
    if (!ensureSessionReady()) {
        alert("请先开始连接");
        return Promise.resolve();
    }

    return fetch("/choice/reset", {
        body: JSON.stringify({
            sessionid: String(document.getElementById("sessionid").value)
        }),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.json();
    }).then(function(payload) {
        if (payload.code !== 0) {
            throw new Error(payload.msg || "choice reset failed");
        }
        renderChoiceState(payload.data);
        choiceState.clientSeq = 0;
    }).catch(function(error) {
        console.error(error);
        $("#choice-note").text("重新开始失败：" + error.message);
    });
}

function requestChoiceSelect(choiceId) {
    if (!ensureSessionReady()) {
        alert("请先开始连接");
        return;
    }

    var selectedButton = $('#choice-options .choice-btn[data-choice-id="' + choiceId + '"]');
    var selectedLabel = selectedButton.find(".choice-text").text().trim() || selectedButton.text().trim();
    $("#choice-options .choice-btn").prop("disabled", true);
    choiceState.clientSeq += 1;
    var requestSeq = choiceState.clientSeq;
    fetch("/choice/select", {
        body: JSON.stringify({
            sessionid: String(document.getElementById("sessionid").value),
            choice_id: choiceId,
            interrupt: true,
            client_seq: requestSeq,
            expected_state_version: choiceState.stateVersion
        }),
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST"
    }).then(function(response) {
        return response.json();
    }).then(function(payload) {
        if (payload.code !== 0) {
            if (payload.code === 409 && payload.data) {
                renderChoiceState(payload.data);
            }
            throw new Error(payload.msg || "choice select failed");
        }
        renderChoiceState(payload.data);
        if (selectedLabel) {
            addChatMessage("选择了：" + selectedLabel, "user");
        }
    }).catch(function(error) {
        console.error(error);
        $("#choice-note").text("选项切换失败：" + error.message);
    }).finally(function() {
        $("#choice-options .choice-btn").prop("disabled", false);
    });
}

function switchPage(page) {
    $(".page-panel").removeClass("active");
    if (page === "conversation") {
        $("#conversation-page").addClass("active");
    } else {
        $("#avatar-selection-page").addClass("active");
    }
}

function bindImageFallback($images) {
    $images.each(function() {
        var image = this;
        image.onerror = function() {
            var fallback = image.getAttribute("data-fallback");
            if (fallback && image.src !== fallback) {
                image.src = fallback;
            }
        };
    });
}

function updateSelectedAvatarDisplay() {
    ensureCurrentAvatar();
    if (!currentAvatar) {
        return;
    }

    $("#selection-note").text("当前选择：" + currentAvatar.name);
    $("#selected-avatar-name").text(currentAvatar.name);
    $("#selected-avatar-description").text(currentAvatar.description);
    $("#side-avatar-name").text(currentAvatar.name);
    $("#selected-avatar-image")
        .attr("src", currentAvatar.image || currentAvatar.placeholder)
        .attr("data-fallback", currentAvatar.placeholder)
        .attr("alt", currentAvatar.name);
    $("#video").attr(
        "poster",
        currentAvatar.image || currentAvatar.placeholder
    );
    bindImageFallback($("#selected-avatar-image"));
    $("#chat-welcome-message").text("系统：当前角色为“" + currentAvatar.name + "”，点击“开始连接”后即可开始对话。");
    $("#conversation-page").toggleClass(
        "camera-offline-mode",
        isCameraAvatar(currentAvatar)
    );
    if (isCameraAvatar(currentAvatar)) {
        $("#choice-note").text("相机人像模式只使用已缓存音频的三选一对话，不会调用实时 TTS。");
    }
}

function setCurrentAvatar(avatarId) {
    var nextAvatar = getAvatarById(avatarId);
    if (!nextAvatar || !nextAvatar.available) {
        return;
    }

    currentAvatar = nextAvatar;
    $(".avatar-card").removeClass("selected");
    $('.avatar-card[data-avatar-id="' + currentAvatar.id + '"]').addClass("selected");
    $("#enter-chat-btn").prop("disabled", false);
    updateSelectedAvatarDisplay();
}

function renderAvatarCards() {
    var container = $("#avatar-grid");
    container.empty();

    avatarCatalog.forEach(function(avatar) {
        var imageSource = avatar.image || avatar.placeholder;
        var disabled = avatar.available ? "" : " disabled aria-disabled=\"true\"";
        var description = avatar.available ? avatar.description : (avatar.reason || "角色资源尚未就绪");
        var card = $(
            '<button class="avatar-card text-start" type="button" data-avatar-id="' + escapeHtml(avatar.id) + '"' + disabled + '>' +
                '<div class="avatar-preview">' +
                    '<img src="' + escapeHtml(imageSource) + '" alt="' + escapeHtml(avatar.name) + '" data-fallback="' + escapeHtml(avatar.placeholder) + '">' +
                    '<span class="avatar-check" aria-hidden="true"><i class="bi bi-check-lg"></i></span>' +
                "</div>" +
                '<div class="avatar-card-body">' +
                    '<div class="avatar-card-title">' +
                        "<h3>" + escapeHtml(avatar.name) + "</h3>" +
                        '<span class="avatar-tag">' + escapeHtml(avatar.badge) + "</span>" +
                    "</div>" +
                    "<p>" + escapeHtml(description) + "</p>" +
                "</div>" +
            "</button>"
        );
        container.append(card);
    });

    bindImageFallback(container.find("img[data-fallback]"));
    var firstAvailable = avatarCatalog.find(function(avatar) {
        return avatar.available && !avatar.cameraCapture;
    }) || avatarCatalog.find(function(avatar) { return avatar.available; });
    if (firstAvailable) {
        setCurrentAvatar(firstAvailable.id);
    }
}

function loadAvatarCatalog() {
    return fetch("/api/avatars")
        .then(function(response) {
            return response.text().then(function(text) {
                var payload;
                try {
                    payload = JSON.parse(text);
                } catch (error) {
                    throw new Error("角色列表接口返回的不是 JSON: " + text.slice(0, 240));
                }

                if (!response.ok || payload.code !== 0) {
                    throw new Error(payload.msg || ("角色列表接口请求失败: " + response.status));
                }

                return payload.data || {};
            });
        })
        .then(function(data) {
            var avatars = Array.isArray(data.avatars) ? data.avatars : [];
            avatarCatalog = normalizeAvatarCatalog(avatars);
            renderAvatarCards();

            if (avatarCatalog.length === 0) {
                $("#selection-note").text("未扫描到可用角色");
            }
        })
        .catch(function(error) {
            console.error(error);
            $("#selection-note").text("角色列表加载失败");
            addChatMessage("角色列表加载失败：" + error.message, "system");
        });
}

function clearRemoteMedia() {
    ["video", "audio"].forEach(function(elementId) {
        var element = document.getElementById(elementId);
        if (element) {
            element.srcObject = null;
        }
    });
}

function closeServerSession(sessionid, beacon) {
    if (!sessionid || sessionid === "0") {
        return;
    }
    var body = JSON.stringify({ sessionid: sessionid });
    if (beacon && navigator.sendBeacon) {
        navigator.sendBeacon(
            "/api/session/close",
            new Blob([body], { type: "application/json" })
        );
        return;
    }
    fetch("/api/session/close", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        keepalive: true
    }).catch(function() {});
}

function negotiate(connection, avatarId) {
    connection.addTransceiver("video", { direction: "recvonly" });
    connection.addTransceiver("audio", { direction: "recvonly" });

    return connection.createOffer()
        .then(function(offer) {
            return connection.setLocalDescription(offer);
        })
        .then(function() {
            return new Promise(function(resolve) {
                if (connection.iceGatheringState === "complete") {
                    resolve();
                } else {
                    var checkState = function() {
                        if (connection.iceGatheringState === "complete") {
                            connection.removeEventListener("icegatheringstatechange", checkState);
                            resolve();
                        }
                    };
                    connection.addEventListener("icegatheringstatechange", checkState);
                }
            });
        })
        .then(function() {
            if (pc !== connection) {
                throw new Error("连接已取消");
            }
            var offer = connection.localDescription;
            var offerPayload = {
                    sdp: offer.sdp,
                    type: offer.type,
                    avatar: avatarId
            };
            if (avatarId === "avatarforcing_camera") {
                offerPayload.capture_id = cameraState.captureId;
            }
            return fetch("/offer", {
                body: JSON.stringify(offerPayload),
                headers: {
                    "Content-Type": "application/json"
                },
                method: "POST"
            });
        })
        .then(function(response) {
            return response.text().then(function(text) {
                var payload;
                try {
                    payload = JSON.parse(text);
                } catch (error) {
                    throw new Error("/offer 返回的不是 JSON: " + text.slice(0, 240));
                }

                if (!response.ok) {
                    throw new Error(payload.msg || ("创建会话失败: " + response.status));
                }

                if (!payload.sdp || !payload.type || !payload.sessionid) {
                    throw new Error("/offer 返回缺少必要字段: " + text.slice(0, 240));
                }

                return payload;
            });
        })
        .then(function(answer) {
            if (pc !== connection) {
                connection.close();
                return;
            }
            if (answer.avatar && answer.avatar !== avatarId) {
                throw new Error(
                    "服务端角色不一致：请求 " + avatarId + "，返回 " + answer.avatar
                );
            }
            document.getElementById("sessionid").value = answer.sessionid;
            return connection.setRemoteDescription(answer).then(function() {
                return requestChoiceInit();
            });
        })
        .catch(function(error) {
            if (pc !== connection) {
                return;
            }
            connection.close();
            pc = null;
            clearRemoteMedia();
            updateConnectionStatus("disconnected");
            $("#stop").hide();
            $("#start").show();
            alert(error.message || String(error));
        });
}

function start() {
    if (!currentAvatar) {
        alert("请先选择一个角色");
        return;
    }
    if (isCameraAvatar(currentAvatar) && !cameraState.captureId) {
        switchPage("selection");
        openCameraCapture();
        return;
    }

    var config = {
        sdpSemantics: "unified-plan"
    };

    if (document.getElementById("use-stun").checked) {
        config.iceServers = [{ urls: ["stun:stun.l.google.com:19302"] }];
    }

    if (pc) {
        pc.close();
        pc = null;
    }
    clearRemoteMedia();
    var connection = new RTCPeerConnection(config);
    var avatarId = currentAvatar.id;
    pc = connection;

    connection.addEventListener("track", function(evt) {
        if (pc !== connection) {
            return;
        }
        if (evt.track.kind === "video") {
            document.getElementById("video").srcObject = evt.streams[0];
        } else {
            document.getElementById("audio").srcObject = evt.streams[0];
        }
    });

    connection.addEventListener("connectionstatechange", function() {
        if (pc !== connection) {
            return;
        }
        if (connection.connectionState === "connected") {
            updateConnectionStatus("connected");
        } else if (connection.connectionState === "connecting") {
            updateConnectionStatus("connecting");
        } else if (["failed", "closed", "disconnected"].indexOf(connection.connectionState) >= 0) {
            updateConnectionStatus("disconnected");
        }
    });

    document.getElementById("start").style.display = "none";
    document.getElementById("stop").style.display = "inline-block";
    updateConnectionStatus("connecting");
    negotiate(connection, avatarId);
}

function stop() {
    var wasConnected = ensureSessionReady();
    var serverSessionId = String(document.getElementById("sessionid").value || "0");
    stopSsvepFlicker();
    ssvepState.targets = [];
    document.getElementById("stop").style.display = "none";
    document.getElementById("start").style.display = "inline-block";
    document.getElementById("sessionid").value = "0";
    updateConnectionStatus("disconnected");
    choiceState.initialized = false;
    choiceState.current = null;
    choiceState.path = [];
    choiceState.stateVersion = 0;
    choiceState.clientSeq = 0;
    choiceState.playbackId = 0;
    $("#choice-answer").text("点击“开始连接”后，可以在这里使用三选一的引导式对话。");
    $("#choice-path").text("当前路径：未初始化");
    $("#choice-options").empty();
    $("#choice-note").text("系统会优先返回文本和选项，并在后台尽量提前准备下一轮音频。");
    updateSsvepStatus();

    var connection = pc;
    pc = null;
    clearRemoteMedia();
    if (connection) {
        connection.close();
    }
    closeServerSession(serverSessionId, false);
    if (wasConnected && isCameraAvatar(currentAvatar)) {
        cameraState.captureId = null;
    }
}

window.onunload = function() {
    stopCameraTracks();
    closeServerSession(
        String(document.getElementById("sessionid").value || "0"),
        true
    );
    if (pc) {
        pc.close();
        pc = null;
    }
};

window.onbeforeunload = function(e) {
    if (pc) {
        pc.close();
        pc = null;
    }
    e = e || window.event;
    if (e) {
        e.returnValue = "确定离开当前页面吗？";
    }
    return "确定离开当前页面吗？";
};

$(document).ready(function() {
    updateConnectionStatus("disconnected");
    loadAvatarCatalog();
    updateSsvepStatus();
    initializeVideoViewport();

    window.LiveTalkingSSVEP = {
        enable: function() {
            setSsvepEnabled(true);
        },
        disable: function() {
            setSsvepEnabled(false);
        },
        setOriginalColor: setSsvepOriginalColor,
        selectTarget: selectChoiceBySsvepTarget
    };

    $("#avatar-grid").on("click", ".avatar-card", function() {
        var avatar = getAvatarById($(this).data("avatarId"));
        if (isCameraAvatar(avatar)) {
            openCameraCapture();
        } else {
            cancelUnclaimedCapture();
            $("#camera-capture-panel").removeClass("active");
            stopCameraTracks();
            setCurrentAvatar(avatar.id);
        }
    });

    $("#camera-take-photo").on("click", captureCameraPhoto);
    $("#camera-upload-image").on("click", function() {
        $("#camera-upload-input").trigger("click");
    });
    $("#camera-upload-input").on("change", selectUploadedAvatarImage);
    $("#camera-retake").on("click", function() {
        resetCameraPhoto();
        setCameraStatus("请重新调整位置，准备好后再次拍照。", false);
    });
    $("#camera-confirm").on("click", uploadCameraCapture);
    $("#camera-cancel").on("click", function() {
        cancelUnclaimedCapture();
        stopCameraTracks();
        resetCameraPhoto();
        $("#camera-capture-panel").removeClass("active");
        $("#enter-chat-btn").prop("disabled", !currentAvatar);
        setCameraStatus("已取消人像录入。", false);
    });

    $("#enter-chat-btn").on("click", function() {
        if (!currentAvatar) {
            return;
        }
        switchPage("conversation");
    });

    $("#back-to-selection-btn").on("click", function() {
        stop();
        switchPage("selection");
    });

    $("#video-size-slider").on("input", function() {
        var value = $(this).val();
        $("#video-size-value").text(value + "%");
        $("#video").css("--video-scale", Number(value) / 100);
    });

    $("#start").on("click", function() {
        start();
    });

    $("#stop").on("click", function() {
        stop();
    });

    $("#choice-init-btn").on("click", function() {
        requestChoiceInit();
    });

    $("#choice-reset-btn").on("click", function() {
        requestChoiceReset();
    });

    $("#ssvep-toggle").on("change", function() {
        setSsvepEnabled(this.checked);
    });

    $("#ssvep-original-color-toggle").on("change", function() {
        setSsvepOriginalColor(this.checked);
    });

    $("#choice-options").on("click", ".choice-btn", function() {
        requestChoiceSelect($(this).data("choiceId"));
    });

    $("#btn_start_record").on("click", function() {
        fetch("/record", {
            body: JSON.stringify({
                type: "start_record",
                sessionid: String(document.getElementById("sessionid").value)
            }),
            headers: {
                "Content-Type": "application/json"
            },
            method: "POST"
        }).then(function(response) {
            if (response.ok) {
                $("#btn_start_record").prop("disabled", true);
                $("#btn_stop_record").prop("disabled", false);
                $("#recording-indicator").addClass("active");
            }
        }).catch(function(error) {
            console.error("Error:", error);
        });
    });

    $("#btn_stop_record").on("click", function() {
        fetch("/record", {
            body: JSON.stringify({
                type: "end_record",
                sessionid: String(document.getElementById("sessionid").value)
            }),
            headers: {
                "Content-Type": "application/json"
            },
            method: "POST"
        }).then(function(response) {
            if (response.ok) {
                $("#btn_start_record").prop("disabled", false);
                $("#btn_stop_record").prop("disabled", true);
                $("#recording-indicator").removeClass("active");
            }
        }).catch(function(error) {
            console.error("Error:", error);
        });
    });

    $("#echo-form").on("submit", function(e) {
        e.preventDefault();
        var message = $("#message").val();
        if (!message.trim()) {
            return;
        }

        postJson("/human", {
                text: message,
                type: "echo",
                interrupt: true,
                sessionid: String(document.getElementById("sessionid").value)
        }).then(function() {
            addChatMessage('已发送播报文本 "' + message + '"', "system");
        }).catch(function(error) {
            console.error(error);
            addChatMessage("播报请求失败：" + error.message, "system");
        });

        $("#message").val("");
    });

    $("#chat-form").on("submit", function(e) {
        e.preventDefault();
        var message = $("#chat-message").val();
        if (!message.trim()) {
            return;
        }

        postJson("/human", {
                text: message,
                type: "chat",
                interrupt: true,
                sessionid: String(document.getElementById("sessionid").value)
        }).catch(function(error) {
            console.error(error);
            addChatMessage("聊天请求失败：" + error.message, "system");
        });

        addChatMessage(message, "user");
        $("#chat-message").val("");
    });

    var mediaRecorder = null;
    var isRecording = false;
    var recognition = null;
    var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "zh-CN";

        recognition.onresult = function(event) {
            var interimTranscript = "";
            var finalTranscript = "";

            for (var i = event.resultIndex; i < event.results.length; i += 1) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                    $("#chat-message").val(interimTranscript);
                }
            }

            if (finalTranscript) {
                $("#chat-message").val(finalTranscript);
            }
        };
    }

    $("#voice-record-btn").on("mousedown touchstart", function(e) {
        e.preventDefault();
        startVoiceRecording();
    }).on("mouseup mouseleave touchend", function() {
        if (isRecording) {
            stopVoiceRecording();
        }
    });

    function startVoiceRecording() {
        if (isRecording) {
            return;
        }

        navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            isRecording = true;
            $("#voice-record-btn").addClass("recording-pulse").css("background-color", "#dc2626");

            if (recognition) {
                recognition.start();
            }
        }).catch(function(error) {
            console.error("Microphone access failed:", error);
            alert("无法获取麦克风权限，请检查浏览器设置。");
        });
    }

    function stopVoiceRecording() {
        if (!isRecording || !mediaRecorder) {
            return;
        }

        mediaRecorder.stop();
        isRecording = false;
        mediaRecorder.stream.getTracks().forEach(function(track) {
            track.stop();
        });

        $("#voice-record-btn").removeClass("recording-pulse").css("background-color", "");

        if (recognition) {
            recognition.stop();
        }

        setTimeout(function() {
            var recognizedText = $("#chat-message").val().trim();
            if (!recognizedText) {
                return;
            }

            postJson("/human", {
                    text: recognizedText,
                    type: "chat",
                    interrupt: true,
                    sessionid: String(document.getElementById("sessionid").value)
            }).catch(function(error) {
                console.error(error);
                addChatMessage("语音发送失败：" + error.message, "system");
            });

            addChatMessage(recognizedText, "user");
            $("#chat-message").val("");
        }, 500);
    }
});
