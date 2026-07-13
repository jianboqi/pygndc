# GeoNDC 编码器使用教程

本文介绍如何使用 `pygndc` 将 GeoTIFF 影像、遥感时序和复合产品编码为
`.gndc` 容器。标准 wheel 同时包含 reader、encoder、native CPU/WGPU runtime；
PyTorch CUDA 与 tiny-cuda-nn 仅在选择 `tcnn_cuda` 时需要。

```bash
pygndc autocompress input.tif -o output.gndc
```

需要调整参数时，先导出 YAML，再按配置执行：

```bash
pygndc autocompress input.tif --save-config config.yaml
pygndc compress -c config.yaml
```

## 1. 后端概览

| Mode | 训练设备 | 外部运行时 | 适用场景 |
|---|---|---|---|
| `native_wgpu` | Vulkan、DX12 或 Metal GPU | 无 | 默认模式；覆盖 NVIDIA、AMD、Intel 与 Apple GPU，失败时回退 native CPU |
| `native_cpu` | CPU | 无 | 无可用 GPU、服务器兼容模式或功能验证 |
| `tcnn_cuda` | NVIDIA CUDA GPU | PyTorch CUDA、CUDA Toolkit、tiny-cuda-nn | NVIDIA 平台上的可选高性能后端 |

三种后端对标准 `hash3d`、无 deformation 的模型写入相同的 GeoNDC v0.9 权重布局。
这类文件的编码后端不限制解码后端：由 `tcnn_cuda` 编码的文件可以使用
`native_wgpu` 或 `native_cpu` 解码，反之亦然。tcnn 专有的 latent/deformation
模型变体不在 native v1 的解码范围内。

native v1 使用 MSE、统一 Adam 学习率、float32 训练和 `norm_mode=3`。它支持
block、importance sampling、空间/时间分块、streaming 和 residual。tcnn 还提供
float16、多种 loss、分离的 hash/MLP 学习率和部分高级模型变体。

## 2. 安装

### 2.1 标准安装

```bash
pip install pygndc
```

安装本地 wheel：

```bash
pip install pygndc-1.0.11-cp39-abi3-win_amd64.whl
```

核心依赖包括 `numpy`、`zstandard`、`rasterio`、`pyyaml`、`tqdm`、
`pydantic` 和 `cryptography`。native 模式不需要 PyTorch、Numba、CUDA Toolkit
或运行时 JIT 编译。

桌面 viewer 是可选组件：

```bash
pip install "pygndc[viewer]"
```

使用本地 wheel 时：

```bash
pip install "pygndc-1.0.11-cp39-abi3-win_amd64.whl[viewer]"
```

验证 native runtime：

```bash
python -c "import pygndc, pygndc._native; print(pygndc.__version__)"
```

### 2.2 可选安装 tcnn CUDA

`tcnn_cuda` 的 pygndc 适配代码已经包含在 wheel 中。用户需要在同一个 Python
环境中自行安装 PyTorch CUDA，并从源码编译 tiny-cuda-nn 的 torch bindings。

系统要求：

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA，建议计算能力 SM 70 或更高 |
| 驱动 | 能够运行所选 PyTorch CUDA 版本 |
| CUDA Toolkit | 提供 `nvcc`，版本应与 PyTorch CUDA 版本兼容 |
| Linux 编译器 | GCC/G++、CMake、Ninja |
| Windows 编译器 | Visual Studio 2019/2022 C++ Build Tools、Windows SDK、CMake、Ninja |

先根据 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装 CUDA wheel，
然后验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

常见 GPU 架构：

| GPU | `TCNN_CUDA_ARCHITECTURES` |
|---|---:|
| RTX 40 系列 | 89 |
| RTX 30 系列 | 86 |
| A100 | 80 |
| RTX 20 系列、T4 | 75 |
| V100 | 70 |

Linux：

```bash
sudo apt-get install -y build-essential git cmake ninja-build
pip install ninja
export TCNN_CUDA_ARCHITECTURES=89
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

如果 CUDA Toolkit 不在默认路径：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Windows 请在 **x64 Native Tools Command Prompt for VS 2022** 中执行：

```bat
conda activate py312
cl
nvcc --version
set TCNN_CUDA_ARCHITECTURES=89
pip install ninja
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```

验证 tcnn：

```bash
python -c "import torch, tinycudann; print(torch.cuda.get_device_name(0))"
```

如果 `tinycudann` 编译失败，请首先检查 `nvcc`、MSVC/GCC、PyTorch CUDA 大版本和
`TCNN_CUDA_ARCHITECTURES`。安装 CPU 版 PyTorch 不能使用 `tcnn_cuda`。

## 3. 编码器授权

读取、解码和 viewer 不需要 encoder 授权。编码操作需要有效的离线授权文件。
如需评估授权或商业许可，请联系 `jianboqi@126.com`。

生成授权请求：

```bash
pygndc license fingerprint
pygndc license request -o request.json
```

收到 `license.dat` 后安装：

```bash
pygndc license install license.dat
pygndc license status
```

默认授权路径：

```text
Windows: %USERPROFILE%\.geondc\license.dat
Linux/macOS: ~/.geondc/license.dat
```

也可以通过 `PYGNDC_LICENSE_FILE` 指定授权文件位置。

## 4. WGPU 设备选择

`native_wgpu` 默认优先选择独立显卡。可在创建第一个 WGPU reader 或 trainer 之前
指定适配器：

```python
import pygndc

for adapter in pygndc.list_wgpu_adapters():
    print(adapter["index"], adapter["preferred"], adapter["backend"],
          adapter["device_type"], adapter["name"])

pygndc.set_wgpu_adapter("NVIDIA")
```

参数是大小写不敏感的名称子串，也可以使用 `"RTX 4080"`、
`"Vulkan: NVIDIA"` 等更具体的值。恢复默认策略：

```python
pygndc.set_wgpu_adapter(None)
```

WGPU device 在进程内共享，因此应在首次打开 `.gndc` 或开始训练前完成选择。
多张同型号显卡应使用 `pygndc.set_wgpu_adapter_index(index)` 精确选择物理卡，
其中 `index` 来自 `list_wgpu_adapters()`。多进程任务也可以分别设置
`PYGNDC_WGPU_ADAPTER_INDEX=0`、`1` 等。

## 5. 命令行编码

### 5.1 自动编码

单幅影像：

```bash
pygndc autocompress input.tif -o output.gndc
```

按日期命名的 GeoTIFF 时序目录：

```bash
pygndc autocompress series/ -o output.gndc
```

大量 GeoTIFF 需要反复训练时，可以先转换成 mmap 友好的 NPY 数据集：

```bash
pygndc tif2npy series/ series_npy/
```

转换结果为一个帧对应一个同名 HWC `.npy` 文件，并包含公共的
`metadata.json`。转换保留原始 dtype 和像素值，同时记录 CRS、仿射变换、
nodata、日期顺序和波段名称。已有且尺寸、dtype 匹配的 NPY 默认跳过；强制重建：

```bash
pygndc tif2npy series/ series_npy/ --overwrite
```

NPY 输入通过配置驱动编码：

```yaml
paths:
  input_path: series_npy/
  output_model: output.gndc

data:
  dataset_mode: npy
  data_scale: 0.0001
  input_valid_range: [0.0, 1.0]
```

`tif2npy` 不执行数值缩放。`data_scale` 和有效值范围仍在编码时应用，空间及时间
分块参数与 TIF 输入完全相同。

仅显示自动规划结果：

```bash
pygndc autocompress input.tif --dry-run
```

可直接覆盖的结构参数：

```bash
pygndc autocompress input.tif -o output.gndc --block-size 3
pygndc autocompress input.tif -o output.gndc --xy-tile-size 1800
pygndc autocompress series/ -o output.gndc --t-tile-size 120
```

### 5.2 配置驱动编码

```bash
pygndc autocompress input.tif --save-config config.yaml
pygndc compress -c config.yaml
```

输入、输出路径可在命令行覆盖：

```bash
pygndc compress -c config.yaml -i other.tif -o other.gndc
```

典型配置：

```yaml
paths:
  input_folder: input.tif
  output_model: output.gndc

data:
  dataset_mode: standard
  n_bands: 10
  data_scale: 1.0
  input_valid_range: [0.0, 1.0]

train:
  mode: native_wgpu          # native_wgpu | native_cpu | tcnn_cuda
  train_dtype: float32       # tcnn_cuda 可使用 float16
  train_loss_type: MSE       # MSE | MSE_RRMSE
  cords_norm_mode: 3
  epochs: 30
  lr: 0.001
  batch_size: 192512
  importance_sampling: true
  imp_alpha: 0.5
  imp_warmup_frac: 0.4
  imp_refresh_epochs: 3
  block_size: [1, 1]

output:
  quantization_bit: 8
  mask_mode: dynamic
  save_residuals: false
  residual_threshold: 0.005

tiled:
  enabled: true
  xy_tile_size: 1800
  t_tile_size: -1
  overlap: 64
  t_overlap: 0
```

`MSE_RRMSE` 对每个波段的 MSE 按该波段目标能量归一化，适合需要平衡低反射率
波段与高反射率波段相对精度的多光谱数据。Native 和 TCNN 都使用整个训练 chunk
的固定波段能量。普通绝对 RMSE 优先时使用 `MSE`；各波段
RRMSE 均衡优先时使用 `MSE_RRMSE`。

切换到 tcnn 只需要修改训练后端；模型和容器参数保持不变：

```yaml
train:
  mode: tcnn_cuda
  train_dtype: float16
```

### 5.3 大图和长时序

空间分块与时间分段都写入同一个 `.gndc` 容器。reader 会自动定位相关 chunk，
并在 overlap 区域进行 feather 混合。

| 参数 | 说明 |
|---|---|
| `block_size` | 一次模型查询输出的像素块，例如 `[3, 3]` |
| `xy_tile_size` | 空间 chunk 边长；`-1` 表示不分块 |
| `t_tile_size` | 每个时间 chunk 的帧数；`-1` 表示不分段 |
| `overlap` | 空间 halo 宽度 |
| `t_overlap` | 相邻时间段的重叠帧数 |
| `importance_sampling` | 提高高误差区域的采样概率 |
| `save_residuals` | 保存稀疏残差轨道，用于约束最大重建误差 |

residual threshold 可以按波段设置，例如 LAI/FPAR：

```yaml
output:
  save_residuals: true
  residual_threshold: [0.5, 0.1]
```

### 5.4 HiGLASS LAI/FPAR

对于以下目录布局：

```text
ROOT/LAI/n042e117/*.tif
ROOT/FPAR/n042e117/*.tif
```

可以使用简化配置：

```yaml
data:
  dataset_mode: higlass_lai_fpar
  tile_name: n042e117
  n_bands: 2
  data_scale: [0.1, 0.004]
  input_valid_range: [0, 10]
```

pygndc 会按共同年份配对 LAI/FPAR 文件，并分别应用 scale 和 nodata。也可以使用
`dual_var_multiband` 加显式 `layout` 描述任意复合产品。

## 6. Python 编码 API

推荐入口：

```python
from pygndc.encoder import autocompress, estimate_config, compress
```

直接编码：

```python
autocompress("input.tif", "output.gndc")
```

先估参再修改：

```python
cfg = estimate_config("input.tif", "output.gndc")
cfg.train_mode = "native_wgpu"
cfg.epochs = 60
cfg.block_size = (3, 3)
cfg.xy_tile_size = 1800
cfg.save_residuals = True
compress(cfg)
```

按 YAML 编码：

```python
compress("config.yaml")
```

底层 `GNDCCompressor` 适合需要自行组织数据加载和训练阶段的应用：

```python
from pygndc.encoder.compressor import GNDCCompressor

compressor = GNDCCompressor(
    train_mode="native_wgpu",
    train_dtype="float32",
)
```

## 7. 性能参考

以下结果用于比较当前实现，不代表所有数据和硬件。总编码时间包含影像读取、训练、
质量评估、量化和容器写入；不包含 residual 与 preview。测试使用 RTX 4080 16 GB、
Python 3.12，从同一幅 10 波段 Sentinel-2 float32 影像中心裁剪，单 chunk，MSE，
`norm_mode=3`，batch 192512，固定 2005 步，int8 权重和 dynamic mask。native 使用
float32；tcnn 使用 float16。

### 7.1 编码速度

| 影像尺寸 | 像素数 | `native_wgpu` | `tcnn_cuda` | native / tcnn |
|---:|---:|---:|---:|---:|
| 512 × 512 × 10 | 0.26 M | 32.01 s | 27.54 s | 1.16× |
| 1024 × 1024 × 10 | 1.05 M | 30.44 s | 26.22 s | 1.16× |
| 2048 × 2048 × 10 | 4.19 M | 32.97 s | 27.15 s | 1.21× |

固定最小训练步数使这组三种尺寸的训练工作量接近，因此总时间不会随像素数线性增加。
生产任务启用自动分块后，总时间主要由 chunk 数、模型规模、输出通道数、block size、
importance refresh 和 residual 扫描共同决定。

### 7.2 解码速度

解码测试使用真实的 20 年 MODIS `.gndc`，单帧 7 波段，热启动后取多次运行的最佳值。
native WGPU 与 tcnn 使用同一张 RTX 4080；native CPU 使用测试主机 CPU。

| 窗口 | `tcnn_cuda` | `native_wgpu` | `native_cpu` |
|---:|---:|---:|---:|
| 256 × 256 × 7 | 15.06 ms | 10.97 ms | 41.11 ms |
| 512 × 512 × 7 | 47.15 ms | 44.89 ms | 131.21 ms |
| 1024 × 1024 × 7 | 159.72 ms | 174.04 ms | 476.34 ms |
| 2400 × 2400 × 7 | 780.19 ms | 868.27 ms | 2482.84 ms |

首次打开还包括容器 metadata、mask、权重上传和 pipeline 初始化，因此冷启动时间不应与
表中的热解码时间直接比较。

## 8. 解码与 Viewer

默认使用 native WGPU：

```python
import pygndc

with pygndc.open("output.gndc") as dataset:
    frame = dataset.read(time=0)
    patch = dataset.read(time=0, window=(0, 0, 512, 512))
```

显式选择后端：

```python
with pygndc.open("output.gndc", mode="native_cpu") as dataset:
    frame = dataset.read(time=0)

with pygndc.open("output.gndc", mode="tcnn_cuda") as dataset:
    frame = dataset.read(time=0)
```

命令行：

```bash
pygndc decompress -i output.gndc -o output_dir --mode native_wgpu
pygndc viewer
pygndc viewer -i output.gndc
```

不带 `-i` 时打开空 viewer，再通过 **File > Open** 选择文件；带 `-i` 时会在窗口
启动后自动打开指定容器。

viewer 默认使用 native WGPU，并支持多 chunk 容器。安装了 tcnn 运行时后，也可以在
Python reader 中显式选择 `tcnn_cuda`。对于标准 hash3d 模型，编码 backend 与解码
backend 无需相同。

## 9. 常见问题

### 没有 NVIDIA GPU 能否编码？

可以。使用 `native_wgpu` 可覆盖 Vulkan、DX12 或 Metal 可见的 GPU；没有可用 GPU 时
会回退 CPU。也可以在 YAML 中设置 `mode: native_cpu`。

### WGPU 选择了集成显卡怎么办？

在首次创建 reader/trainer 前执行 `pygndc.set_wgpu_adapter("NVIDIA")`、
`pygndc.set_wgpu_adapter("AMD")` 或更具体的显卡名称。
多张同型号显卡使用 `pygndc.set_wgpu_adapter_index(index)`。

### `tcnn_cuda` 提示缺少模块怎么办？

确认当前 Python 环境能够同时导入 `torch` 和 `tinycudann`，并且
`torch.cuda.is_available()` 为 `True`。tiny-cuda-nn 必须针对当前 PyTorch/CUDA 环境编译。

### Viewer 提示缺少 matplotlib 或 Pillow

安装 viewer extra：

```bash
pip install "pygndc[viewer]"
```

### 如何选择 block size？

`[1, 1]` 提供最直接的逐像素模型；较大的 block 可减少坐标查询数量，但会增加输出层
宽度。建议先使用 `autocompress --dry-run` 或导出配置后在代表性 tile 上比较速度和质量。
