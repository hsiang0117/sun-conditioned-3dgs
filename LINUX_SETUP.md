# Linux / 服务器环境

服务器目录为 `/myfiles/projects/sun-conditioned-3dgs`。项目使用独立 `.venv`；PyTorch 和三个 CUDA 扩展均安装在该环境中，不使用其他项目的 `site-packages`。

本次配置：Ubuntu 24.04、RTX 4090（sm_89）、Python 3.12.14、CUDA Toolkit 12.8.93、GCC/G++ 13.3。Python 解释器来自已有的 `/opt/conda`，与其他项目共享解释器和系统 CUDA SDK，Python 包独立安装。运行与构建依赖使用 `requirements-runtime.txt` 和 `requirements-build.txt` 的固定版本。

## 常规安装

```bash
git clone https://github.com/hsiang0117/sun-conditioned-3dgs.git
cd sun-conditioned-3dgs
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-runtime.txt -r requirements-build.txt
git submodule update --init submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
git -C submodules/diff-gaussian-rasterization submodule update --init --recursive
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 bash tools/build_cuda_extensions.sh
```

构建需要 Python 开发头文件、GCC/G++ 和完整 CUDA Toolkit。其他 GPU 应设置其对应的 `TORCH_CUDA_ARCH_LIST`。无需初始化 SIBR viewer 子模块。

## 本次安装的缓存复用

服务器现有的 CUDA/PyTorch Linux wheel 位于：

```text
/myfiles/projects/Vol3DGS/dependency-cache/wheels
```

核对该目录中的包后，本次先将其中的 torch、torchvision 及其 CUDA 运行库依赖离线安装到新环境，再安装其余运行和构建依赖，避免重新下载大包：

```bash
cache=/myfiles/projects/Vol3DGS/dependency-cache/wheels
.venv/bin/python -m pip install --no-index --no-deps "$cache"/*.whl
.venv/bin/python -m pip install -r requirements-runtime.txt -r requirements-build.txt
```

这条缓存命令仅针对本次核对过的目录，不能用于任意混合版本的 wheel 文件夹。本次还从服务器已有原版 3DGS checkout 复制了固定提交的 Git 对象，随后恢复各子模块的正式上游 URL；新 checkout 不依赖旧仓库的 Git alternates。

Linux 构建使用原始 PyTorch 头文件，**不执行 Windows 的 `patch_torch_header.py`**。编译脚本验证 Python/torch 来自本项目环境，检查子模块提交并记录头文件 SHA256；日志和环境记录保存在 `temporary-build/`。构建完成后运行 9 项数值、梯度及保存恢复检查和 `pip check`。

## 正式重光照实验入口

主对比使用 env-on 数据集，评估目标与 Cloud-GS 第二阶段一致。服务器的数据目录为 `/myfiles/data/CloudDatasetUniform_envon`：

```bash
.venv/bin/python train.py -s /myfiles/data/CloudDatasetUniform_envon --eval --resolution 1 --data_device cpu --disable_viewer
.venv/bin/python render.py -m output/<run> --skip_train
.venv/bin/python metrics.py -m output/<run>
.venv/bin/python tools/eval_test_groups.py output/<run>
```

默认训练 30000 步，输出为时间戳目录，每 1000 步保存固定训练机位/太阳的预览图。环境安装和数值检查不代表已完成该正式训练。

图像读取默认使用 Cloud-GS 的按需解码、CPU uint8 缓存、用后释放和后台预取方式，保留本仓库原有的 PIL 缩放和 RGBA alpha。`--image_cache_max 0` 缓存全部已读帧；内存紧张时可设为 `256` 等正整数，通过 LRU 限制缓存张数。1308/152 张 1024×1024 RGBA 图像的 uint8 缓存最多约 5.70 GiB，不含模型及运行开销。服务器容器的内存限额应读取 `/sys/fs/cgroup/memory.max`；宿主机 `free -h` 不是容器可用额度。这些 Python 改动无需重新编译 CUDA 扩展。

已有训练可使用相同数据/参数、原输出目录和 `--start_checkpoint output/<run>/chkpnt20000.pth` 续训到默认的 30000 步。checkpoint 包含模型与优化器，不包含原采样器和 RNG 状态；续训不是未中断训练轨迹的逐位恢复。

## 本次验证（2026-10-02）

- 三个 CUDA 扩展均在本项目环境中编译安装，使用 sm_89；CUDA 源码与固定提交保持一致。
- 9 项检查全部通过，覆盖零增量图像/几何梯度、条件网络梯度、增密和裁剪、PLY/训练 checkpoint 恢复、太阳编码与分组完整性。
- fused-SSIM 在 GPU 上的前向及反向检查通过；`pip check` 没有依赖冲突。
- torch、torchvision、三个扩展的实际导入路径均位于本项目 `.venv` 内。两个 PyTorch 头文件的 SHA256 与缓存 Linux wheel 内的原件一致。
- 训练、渲染和固定太阳导出命令的入口检查正常。env-on 目录、1308/152 划分和 4 个未见太阳 ID 已核对。正式训练结果另行记录在对应的时间戳输出目录中。
- 迁移图像读取优化后，本机和服务器的全部 17 项测试通过，包括原有 9 项条件模型检查及新增 8 项图像缓存/预取检查；CUDA 预取后的 GT、损失和梯度与旧实现逐值一致。
- 服务器在 1024×1024 上遍历全部 1460 帧，缓存为 5.70 GiB，进程 RSS 为 7.09 GiB、容器总占用为 8.56 GiB；这些是加载验证的实测值，不是训练峰值。场景构建未解码像素，抽检的 10 张真实 GT/alpha 与旧 PIL 读取逐值一致。训练容器的实际内存上限为 56 GiB。
- env-on 正式训练 `output/20261002_104519` 已从 20000 步 checkpoint 续训到 30000 步，152 张测试渲染、整体指标和分组指标均已完成，没有新增 OOM。续训 10000 步连同预览和训练末尾评估耗时 265.21 秒，进程内存峰值为 8.25 GiB；全图测试指标为 PSNR 31.9052 dB、SSIM 0.97265、LPIPS 0.05840。失败那次的日志保存在 `recovery/attempt_1`，这些时间仅覆盖续训部分。
