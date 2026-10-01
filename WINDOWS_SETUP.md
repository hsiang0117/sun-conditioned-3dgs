# Windows 环境与 CUDA 构建

本项目使用独立 `.venv`，不向原版 3DGS 或 Cloud-GS 的环境安装包。当前本机为 RTX 5060 Laptop（sm_120，8GB），CUDA Toolkit 12.8，MSVC 14.44，Python 3.12。运行依赖固定在 `requirements-runtime.txt`，构建前端版本固定在 `requirements-build.txt`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt -r requirements-build.txt
git submodule update --init submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
git -C submodules/diff-gaussian-rasterization submodule update --init --recursive
.\tools\build_cuda_extensions.ps1 -CudaArch 12.0 -Toolset 14.44 -Jobs 4
```

PyTorch 和 torchvision 也可以直接安装缓存中的对应 `.whl`，之后再安装运行依赖，避免重新下载大文件。wheel 必须与 Python 版本、Windows 平台及记录的 torch 版本一致。不要复制其他项目的 `site-packages`，也不要从旧缓存任选一个同名 CUDA 扩展包。

## 两个已验证的 PyTorch 头文件兼容修改

`tools/patch_torch_header.py` 只接受本项目 `.venv` 中的 `torch==2.11.0+cu128`，对以下文件先验证 SHA256，再备份原件并修改。未知版本直接停止。

- `torch/csrc/dynamo/compiled_autograd.h`：沿用之前几个仓库验证过的 Windows/NVCC 构建处理，注释 `std::string` 到 `at::StringType::get()` 的模板分支。它涉及 compiled-autograd 模板类型映射，并非通用的等价重写；本项目只使用常规 eager autograd，没有启用 compiled autograd。
- `c10/cuda/CUDACachingAllocator.h`：把构造函数的 `bool small` 参数改名为 `is_small_flag`，避免 Windows 的 `small` 宏干扰。此项不改变行为。

兼容脚本保留原件于 `environment-backups/`，可恢复：

```powershell
.\.venv\Scripts\python.exe -I tools\patch_torch_header.py --restore
```

构建脚本调用 `vcvarsall.bat` 后明确设置 `DISTUTILS_USE_SDK`、`CUDA_HOME`、`TORCH_CUDA_ARCH_LIST`、`MAX_JOBS`，使用中文 Windows 的代码页 936，分别构建三个固定提交的子模块；随后执行数值/梯度检查和 `pip check`。请在有正常控制台的 PowerShell 中运行，代码页设置失败时会停止。

CUDA 扩展固定提交：

| 扩展 | 提交 |
| --- | --- |
| diff-gaussian-rasterization | `0de1692522851a0ed47baeb69b7c63f99bbd9898` |
| simple-knn | `86710c2d4b46680c02301765dd79e465819c8f19` |
| fused-ssim | `1272e21a282342e89537159e4bad508b19b34157` |
| rasterizer 的 GLM | `5c46b9c07008ae65cb81ab79cd677ecc1934b903` |

条件颜色由 Python/PyTorch 生成，再传入现有 `colors_precomp`，没有修改这些子模块的 CUDA kernel。

## 环境文件的用途

- `.venv/`：本项目运行环境。
- `environment-backups/`：需保留的原始头文件。
- `dependency-cache/`：供以后离线安装复用的包缓存；可以为空。
- `temporary-build/`：构建日志、环境核查信息和临时测试运行日志，可在核查后删除。

这几个目录均不提交到 Git。完整实验的输出仍在 `output/`。若本机内存不足，可以用 `--resolution 4` 做流程验证；正式对比必须恢复一致的图像分辨率。此处固定的是已验证的 Windows 配置，Linux 服务器应单独准备其对应版本的依赖，不能套用 Windows wheel 或头文件补丁。
