# AvatarForcing 最终 Demo：环境创建与运行指令

本文面向一台全新的 Linux 服务器，目标是复现本仓库固定的
AvatarForcing Talking-only Demo。模型权重不在 Git 仓库中，安装脚本会从官方来源下载并进行 SHA-256 校验；520 个离线语音缓存已随仓库发布。

## 1. 系统前提

- Linux x86_64（推荐 Ubuntu 22.04 或 24.04）；
- NVIDIA GPU 和可用的 NVIDIA 驱动；
- Git、Conda（Miniconda 或 Miniforge）；
- 能访问 GitHub、Google Drive、Hugging Face 和模型权重下载站点；
- 建议预留至少 20 GB 磁盘空间。

先检查基础工具：

```bash
nvidia-smi
git --version
conda --version
df -h .
```

`nvidia-smi` 必须能够显示 GPU 和驱动信息。安装脚本使用 Python 3.10，以及
PyTorch 2.5.0 / torchvision 0.20.0 / torchaudio 2.5.0 的 CUDA 12.4 wheel；系统无需单独安装 CUDA Toolkit，但 NVIDIA 驱动必须与该运行时兼容。

## 2. 推荐：一键创建环境、下载权重并校验

```bash
git clone https://github.com/pmy6666/Live_demo.git
cd Live_demo
bash scripts/avatarforcing/prepare_final_demo.sh
```

该命令依次完成：

1. 在仓库同级创建 `envs/livetalking` Conda 环境；
2. 安装固定版本的 Python 包；
3. 克隆固定提交的 AvatarForcing 源码并应用本仓库补丁；
4. 下载所有模型权重并校验哈希；
5. 校验离线语音缓存和最终运行文件。

脚本可重复执行：已经正确下载的文件会被复用或重新校验。

## 3. 自定义环境目录

默认 Python 路径为：

```text
<Live_demo 仓库的父目录>/envs/livetalking/bin/python
```

若需要将环境放在其他目录，在所有相关命令中设置同一个变量：

```bash
export LIVETALKING_ENV_PREFIX=/opt/envs/livetalking
bash scripts/avatarforcing/prepare_final_demo.sh
```

后续运行时使用：

```bash
/opt/envs/livetalking/bin/python start_avatarforcing.py
```

## 4. 分步安装指令

如果需要分别执行环境安装和权重下载：

```bash
git clone https://github.com/pmy6666/Live_demo.git
cd Live_demo
bash scripts/avatarforcing/setup_env.sh
bash scripts/avatarforcing/download_weights.sh
```

仅检查文件、补丁、权重和缓存，不检查 GPU：

```bash
../envs/livetalking/bin/python start_avatarforcing.py \
  --validate-only --files-only
```

检查 PyTorch 是否能够使用 CUDA：

```bash
../envs/livetalking/bin/python -c \
  'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
```

执行完整启动前检查：

```bash
../envs/livetalking/bin/python start_avatarforcing.py --validate-only
```

## 5. 运行 Demo

在服务器中执行：

```bash
cd /path/to/Live_demo
../envs/livetalking/bin/python start_avatarforcing.py
```

如果仓库位于 `/home/qianustb/Live_demo`，对应命令就是：

```bash
cd /home/qianustb/Live_demo
/home/qianustb/envs/livetalking/bin/python start_avatarforcing.py
```

服务器本机访问地址：

```text
http://127.0.0.1:8010/dashboard.html
```

通过 SSH 使用该服务器时，在自己的本地电脑另开终端运行：

```bash
ssh -N -L 8010:127.0.0.1:8010 qianustb@10.23.39.169
```

保持隧道终端开启，然后用本地浏览器访问：

```text
http://127.0.0.1:8010/dashboard.html
```

注意：SSH 隧道可以转发网页和信令 TCP 流量。若网页可打开但视频无法建立连接，还需要允许 WebRTC UDP 流量，或使用 TURN、SRS、VPN 等网络方案。

## 6. 国内网络与常见问题

如果 Hugging Face 无法访问，可在下载权重前设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/avatarforcing/download_weights.sh
```

常见问题：

- `conda is required`：先安装 Miniconda/Miniforge，并重新打开终端；
- `CUDA available: False`：检查 `nvidia-smi`、宿主机驱动以及容器的 GPU 映射；
- 端口 8010 被占用：停止占用该端口的进程，再启动最终 Demo；
- 权重下载中断：重新执行下载脚本，完成后仍会逐个校验哈希；
- `AvatarForcing has local changes`：安装目录中的上游源码被修改。备份所需改动后重新克隆干净仓库；
- 浏览器页面能打开但没有画面：优先检查浏览器控制台、服务器防火墙和 WebRTC UDP 连通性。

更完整的版本固定、运行文件清单和发布说明见
[AvatarForcing 最终 Demo 发布与复现](avatarforcing_final_release.md)。
