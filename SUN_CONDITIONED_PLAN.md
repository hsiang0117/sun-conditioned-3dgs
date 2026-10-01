# Sun-conditioned 3DGS 实施方案

状态：2026-10-01，已建立独立仓库并分析实现入口，尚未加入条件外观网络，也未开始训练。

## 仓库与基线

- 研究仓库：https://github.com/hsiang0117/sun-conditioned-3dgs
- 官方上游：https://github.com/graphdeco-inria/gaussian-splatting
- 现有对比仓库：https://github.com/hsiang0117/gaussian-splatting
- 本机目录：`D:\PythonProjects\sun-conditioned-3dgs`
- 起始提交：`82c24968dda8933575019189eec5fcd568d638a8`。

GitHub CLI 的第二次 fork 操作重命名了已有 fork，因此已恢复原仓库名称，并另建保留完整 main 提交历史的独立仓库。本项目在 GitHub 上不带 fork 标记，仍通过 upstream 跟踪官方代码。原始 LICENSE.md 和版权声明保留。

起始版本与当前本机原版 3DGS 一致。相对该版本记录的官方 upstream/main，已有四项适配：pip 环境配置、CUDA 扩展 cstdint 构建修复、时间戳输出目录、带扩展名的图像路径及 Pillow uint8 修复。没有引入 Cloud-GS 的物理模型或 CUDA 渲染修改。

## 对比方法的定位

定义一个自建的 Sun-conditioned 3DGS 基线：在所有太阳方向之间共享高斯位置、尺度、旋转和 opacity，只让颜色外观依赖太阳方向。共享表示在训练中仍正常优化，不是冻结现有几何进行诊断。

它用已知光照条件学习图像外观，不显式计算消光、自阴影、相函数或多重散射。论文中应单独标为对原版 3DGS 的条件扩展，并说明实现和模型容量，不能把它称为原版 3DGS 或已发表的方法。实验用于判断：仅增加太阳条件，能否解释数据驱动方法的重光照表现。

## 推荐第一版：条件 SH 增量

设 d_i 为从相机中心指向高斯中心的单位方向，保持原版 SH 的方向约定；s 为世界坐标系中指向太阳的单位向量。

保留原版每个高斯的视角 SH 系数 A_i，新增 16 维可学习特征 z_i。共享小型 MLP 接收 [z_i, SH_2(s)]，输出与原版 SH 系数同形状的增量 DeltaA_i(s)：

    A_i(s) = A_i + MLP_theta(z_i, SH_2(s))
    c_i(d_i, s) = max(0, 0.5 + sum_k A_i,k(s) Y_k(d_i))

SH_2(s) 包括 0 至 2 阶，共 9 项；网络使用两层 32 宽隐藏层，ReLU 激活，输出默认 3 阶视角 SH 的 16 x 3 个系数。网络输出要按照 GaussianModel.get_features 的 [N, K, 3] 布局恢复，并尊重 active_sh_degree。

- 太阳必须用连续方向编码，不能用 time_index 的离散 embedding 替代，否则未见太阳方向没有直接可用的条件。
- 输出层权重和偏置初始化为零；z_i 采用小幅随机初始化。增量为零时回到原版 SH 表达，可用原版模型进行一致性验证。
- 太阳影响全部视角 SH 系数，不只添加一个全局亮度或 RGB 偏移，因此能表达太阳与观察方向的联合变化。
- 该网络和 SH 仍具有有限容量，对尖锐阴影与强前向散射的拟合能力须由实验判断。
- 第一版维持原版 L1 + DSSIM 损失；LPIPS 仅用于最终评估。

## 不修改 CUDA 的实现入口

原版 rasterizer 已接收 colors_precomp，backward 也返回 grad_colors_precomp。可在 Python 中生成条件 SH、求值为 [N, 3] 颜色，再通过 colors_precomp 渲染。

必须保持 autograd 链，不得 detach 条件颜色，也不得使用 torch.no_grad 包住训练中的颜色求值。视角方向依赖高斯位置，Python SH 求值产生的位置梯度应保留。

条件颜色路径使用普通 colors_precomp 调用，不进入 separate_sh 的 dc/shs 分支。当前 renderer 在 override_color 与 separate_sh 同时启用时可能访问未赋值的 dc，需要在这一入口明确处理。

修改范围：

| 文件 | 工作 |
| --- | --- |
| `scene/dataset_readers.py` | 加载、验证并归一化 sun_direction；生成包含相机和太阳信息的唯一帧标识。 |
| `scene/cameras.py`、`utils/camera_utils.py` | 将太阳方向传入 Camera，并写入相机元数据。 |
| `scene/gaussian_model.py` | 新增 z_i、共享条件网络、优化器及完整的保存/恢复逻辑。 |
| `gaussian_renderer/__init__.py` | 生成条件 SH、Python SH 求值，通过 colors_precomp 进入现有 rasterizer。 |
| `train.py`、`arguments/__init__.py` | 条件网络优化器、可复现配置、固定训练预览及每 1000 步保存图像。 |
| `render.py`、评估与导出工具 | 逐帧太阳条件、分组评估，以及给定太阳方向的标准 PLY 导出。 |

z_i 属于逐高斯参数，需纳入 clone、split、prune 和 Adam 状态同步。现有 _prune_optimizer / cat_tensors_to_optimizer 假设所有参数组首维都是高斯数量；共享 MLP 不能直接加入这个优化器，应使用独立优化器。

训练 checkpoint 要保存两套优化器及网络配置。正式模型除原有 point_cloud.ply 外，按迭代保存 sun_conditioning.pt，包含 z_i、网络权重和结构配置，并验证与 PLY 的点数、顺序一致。

固定一个太阳方向后，A_i(s) 就是一套标准视角 SH 系数，可另导出 baked PLY 给原版 viewer 使用。这个 PLY 只对应选定的太阳；动态调太阳需要条件模型文件和支持条件网络的渲染入口。

## 当前本机数据核查

读取的实际文件为 `D:\dataset\CloudDatasetUniform\transforms_train.json` 和 `transforms_test.json`：

- 训练 1308 帧，56 个太阳方向。
- 测试 152 帧；太阳 ID 7、22、37、52 在训练中完全不存在，共 96 帧。
- 剩余 56 帧属于已见太阳下未训练过的相机/太阳组合；这些相机位置全部在训练集中出现过，不能称为全新机位。
- 所有帧均含 sun_direction，已有 points3d.ply。
- sun_direction 已转换为 OpenGL 世界坐标，+Y 向上，向量指向太阳，无需再次进行 UE 轴变换。
- 不同 camXX 下重复出现同一图像 stem，须使用完整相对路径或 camera_index/time_index 作为唯一帧标识。

Zenith 的太阳方向只有一个，适合核对静态退化行为，不能训练或评估太阳条件泛化。

## 训练与评估安排

1. 实现数据链、条件 SH、增密参数同步和 checkpoint；建立独立 .venv，复用原版环境的版本及 CUDA 构建方式，不能安装到 Cloud-GS 的 .venv 中。
2. 验证零增量分支与原版图像/梯度一致；检查方向正负号、数据 split、增密/裁剪后点数和参数对应，以及保存/加载图像一致。
3. 在 Uniform 上跑短程 1000 步，比较原版与条件版本的每步耗时、峰值显存、点数和损失。新增 16 维特征每 10 万点的 FP32 参数约 6.4 MB，连同梯度和 Adam 两个状态约 25.6 MB；这不包含网络激活和临时 SH 张量，完整开销须实测。
4. 短程结果和开销正常后，再按原版 30000 步设置完整训练。使用 --eval、--resolution 1、固定黑背景，关闭逐图曝光拟合，保持原版几何优化和增密设置；预览使用固定训练机位/太阳，不根据测试图调参。
5. 保持同一初始化点云、图像分辨率、训练/测试 split 和评估协议。分别报告 96 帧未见太阳、56 帧已见太阳下新组合及总体结果；同时报告全图与现有 VDB mask 的云区域 PSNR/SSIM/LPIPS，以及训练时间、显存、点数、模型大小。

先用这一个明确的条件扩展建立基线。若训练集拟合和未见太阳结果出现不同趋势，再决定调整网络容量或太阳编码；不能预先把效果归因于物理表示的优势。

## 本轮完成边界

本轮只建立仓库、保留原版基线并记录以上方案。尚未创建新环境、编译扩展、改动训练算法或启动训练。
