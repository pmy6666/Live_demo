# AvatarForcing 最终 Demo 发布与复现

如果只需在全新服务器上创建环境并运行，请直接使用
[环境创建与运行指令](avatarforcing_environment_setup.md)。

本文只描述以下最终入口，不包含 EchoMimicV3、LatentSync、MuseTalk 或 GPT-SoVITS 在线服务：

```bash
cd /home/qianustb/LiveTalking
/home/qianustb/envs/livetalking/bin/python start_avatarforcing.py
```

## 1. 固定运行契约

- Python 3.10，PyTorch 2.5.0 + CUDA 12.4 wheel。
- 单 GPU、单会话、25 FPS、16 kHz PCM、WebRTC 端口 8010。
- 默认角色 `avatarforcing_male`，并支持内置女性角色和摄像头上传角色。
- 最终入口使用独立的 `data/avatar_profiles_avatarforcing.json`，不会加载或检查其他模型的角色与缓存。
- Talking-only 推理固定为 `seed=20`、`nfe=10`、`a_cfg_scale=2.0`、`u_cfg_scale=0.0`、`pad_ratio=1.0`、KV cache 开启。
- 对话声音来自 `cache/avatarforcing_tts/{female,male}` 的 520 个离线 PCM 文件，运行 Demo 不需要启动 GPT-SoVITS。

完整运行文件清单见 `release/avatarforcing/runtime-files.txt`，版本、权重和缓存校验值见 `release/avatarforcing/manifest.json`。

## 2. 全新机器安装

系统前提：Linux x86_64、Conda、Git、NVIDIA 驱动，以及能运行 CUDA 12.4 PyTorch wheel 的 NVIDIA GPU。系统还需要允许浏览器与服务器建立 WebRTC UDP 连接。

```bash
git clone https://github.com/pmy6666/Live_demo.git LiveTalking
cd LiveTalking
bash scripts/avatarforcing/prepare_final_demo.sh
```

该一键脚本包含环境创建、权重下载与文件校验。若希望分步执行，可先运行
`scripts/avatarforcing/setup_env.sh`，再按下一节下载权重。

默认环境位置是仓库同级的 `envs/livetalking`。如需其他位置：

```bash
LIVETALKING_ENV_PREFIX=/opt/envs/livetalking \
  bash scripts/avatarforcing/setup_env.sh
```

安装脚本会克隆 AvatarForcing 官方仓库的固定提交
`507e3046ad539c938e0412ac7a74cf6e3ba2a61c`，再应用本仓库保存的 Talking-only 与流式推理补丁。`AvatarForcing/` 是构建产物，不提交到主仓库。

## 3. 下载模型权重

```bash
bash scripts/avatarforcing/download_weights.sh
```

脚本下载并校验：

- AvatarForcing `motion_autoencoder.pth` 和 `flow_transformer.pth`；
- 固定提交 `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` 的
  `facebook/wav2vec2-base-960h` 最小运行文件；
- face-alignment 使用的 S3FD 和 2DFAN4 权重。

这些权重均被 `.gitignore` 排除，不会进入 Git 历史。

## 4. 校验离线语音缓存

520 个离线 PCM 文件直接随仓库发布，不属于模型权重，无需额外下载。克隆后执行：

```bash
../envs/livetalking/bin/python \
  scripts/avatarforcing/build_avatarforcing_tts_cache.py verify
```

如需制作独立备份包：

```bash
bash scripts/avatarforcing/package_runtime_cache.sh
```

产物位于 `release/artifacts/avatarforcing_tts_cache_v1.tar.gz`，该备份包不会重复提交到 Git。

## 5. 启动前校验与运行

只检查文件、补丁产物、权重和缓存，不检查 GPU：

```bash
../envs/livetalking/bin/python start_avatarforcing.py \
  --validate-only --files-only
```

检查文件和 CUDA：

```bash
../envs/livetalking/bin/python start_avatarforcing.py --validate-only
```

启动最终 Demo：

```bash
../envs/livetalking/bin/python start_avatarforcing.py
```

运行不加载真实模型的回归测试：

```bash
PYTHONPATH=. ../envs/livetalking/bin/python -m unittest discover \
  -s test -p 'test_avatarforcing_*.py'
PYTHONPATH=. ../envs/livetalking/bin/python -m unittest discover \
  -s test -p 'test_camera_*.py'
```

本机浏览器访问：

```text
http://127.0.0.1:8010/dashboard.html
```

通过 SSH 访问服务器时，在本地电脑运行：

```bash
ssh -N -L 8010:127.0.0.1:8010 qianustb@10.23.39.169
```

然后在本地浏览器访问同一个 `http://127.0.0.1:8010/dashboard.html`。SSH 隧道只转发 TCP 页面和信令；如果页面能打开但无视频，还需要打通 WebRTC UDP，或部署 TURN/SRS/VPN。

## 6. 发布检查

发布源码时只选择 `release/avatarforcing/runtime-files.txt` 中的父仓库文件以及这些文件的实际依赖。不要提交以下内容：

- `AvatarForcing/` 克隆目录；
- `AvatarForcing/pretrained_dir/` 权重；
- `.runtime/` face-alignment 权重；
- `release/artifacts/`；
- 其他模型目录、实验输出、日志和 Python 环境。

公开发布前还应确认男女角色图片、参考音频和缓存声音素材具备公开分发授权。
