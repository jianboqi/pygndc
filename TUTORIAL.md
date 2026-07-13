# pygndc Tutorial: Reading and Analyzing `.gndc` Files

This tutorial covers installation, backend selection, metadata, spatial and temporal
queries, analysis helpers, export, and the desktop viewer. For the container specification, see
[FORMAT_SPECIFICATION.md](FORMAT_SPECIFICATION.md).

## 1. Installation

Install the standard package:

```bash
pip install pygndc
```

The standard wheel contains the native encoder and decoder. Native execution does not
require PyTorch, Numba, a CUDA Toolkit, or runtime compilation. `native_wgpu` uses a
Vulkan, DX12, or Metal adapter and falls back to `native_cpu` if WGPU is unavailable.

Install the optional viewer dependencies with:

```bash
pip install "pygndc[viewer]"
```

### Optional tcnn CUDA runtime

The `tcnn_cuda` integration is included in pygndc, but its external runtime is not.
To use it, install a CUDA-enabled PyTorch build and compile tiny-cuda-nn's torch
bindings for the same Python/CUDA environment. Detailed Linux and Windows instructions can be reffered to [GitHub - NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework · GitHub](https://github.com/nvlabs/tiny-cuda-nn).

Verify an optional tcnn installation:

```bash
python -c "import torch, tinycudann; print(torch.cuda.is_available())"
```

## 2. Inference Backends

| Mode          | Runtime                | Device                     | Notes                                                 |
| ------------- | ---------------------- | -------------------------- | ----------------------------------------------------- |
| `native_wgpu` | Rust/WGPU              | Vulkan, DX12, or Metal GPU | Default; prefers a discrete GPU and falls back to CPU |
| `native_cpu`  | Rust                   | CPU                        | Portable fallback                                     |
| `tcnn_cuda`   | PyTorch + tiny-cuda-nn | NVIDIA CUDA GPU            | Optional user-installed runtime                       |

List visible WGPU adapters and select one before the first WGPU reader is created:

```python
import pygndc

for adapter in pygndc.list_wgpu_adapters():
    print(adapter["index"], adapter["backend"], adapter["device_type"],
          adapter["preferred"], adapter["name"])

pygndc.set_wgpu_adapter("NVIDIA")
```

Use `pygndc.set_wgpu_adapter(None)` to restore the default discrete-GPU-first policy.
On systems with multiple identical GPUs, select the physical adapter by index before
creating the first reader or trainer:

```python
pygndc.set_wgpu_adapter_index(3)
```

For separate worker processes, set `PYGNDC_WGPU_ADAPTER_INDEX=0`, `1`, and so on.

## 3. Opening a Container

Use a context manager so model and container resources are released promptly:

```python
import pygndc

with pygndc.open("satellite.gndc") as dataset:
    print(dataset.info())
    frame = dataset.read(time=0)
```

Select a backend explicitly when required:

```python
with pygndc.open("satellite.gndc", mode="native_cpu") as dataset:
    frame = dataset.read(time=0)

with pygndc.open("satellite.gndc", mode="tcnn_cuda") as dataset:
    frame = dataset.read(time=0)
```

Single-chunk and multi-chunk containers use the same high-level API. Chunk routing,
spatial overlap blending, temporal segment selection, masks, and residual correction
are handled by `GNDCDataset`.

## 4. Metadata

```python
with pygndc.open("satellite.gndc") as dataset:
    print(dataset.shape)             # (T, H, W, C)
    print(dataset.n_timestamps)
    print(dataset.n_bands)
    print(dataset.height, dataset.width)
    print(dataset.timestamps)
    print(dataset.crs)
    print(dataset.transform)
    print(dataset.bounds)
    print(dataset.resolution)
    print(dataset.band_names)
    print(dataset.band_wavelengths)
    print(dataset.has_mask)
    print(dataset.has_residuals)
    print(dataset.metrics())
```

`metrics()` returns physical-unit quality metrics recorded by the encoder when they
are available, including per-band RMSE, MAE, maximum error, and PSNR.

## 5. Reading Data

The high-level API separates observed and synthesized time semantics:

| Method          | Time semantics                | Return shape                     |
| --------------- | ----------------------------- | -------------------------------- |
| `read()`        | Stored observation            | area `(H, W, C)` or point `(C,)` |
| `interpolate()` | Continuous model time         | area `(H, W, C)` or point `(C,)` |
| `series()`      | All stored times at one point | `(T, C)`                         |

### Observed frames

```python
with pygndc.open("satellite.gndc") as dataset:
    frame = dataset.read(time=0)
    frame = dataset.read(time="2024-06-15")

    # window=(column offset, row offset, width, height)
    patch = dataset.read(time=0, window=(100, 200, 512, 512))

    # bbox=(minimum x, minimum y, maximum x, maximum y)
    region = dataset.read(time=0, bbox=(116.3, 39.8, 116.5, 40.0))

    pixel = dataset.read(time=0, rowcol=(140, 200))
    pixel = dataset.read(time=0, latlng=(116.825, 40.486))
    rgb = dataset.read(time=0, bands=[2, 1, 0])
```

`read()` accepts an integer index or an exact stored timestamp. Use `interpolate()`
for dates between observations.

### Continuous-time reconstruction

```python
with pygndc.open("satellite.gndc") as dataset:
    frame = dataset.interpolate("2024-06-15 12:00:00")
    midpoint = dataset.interpolate(0.5, window=(0, 0, 512, 512))
```

A float time is interpreted as normalized model time in `[0, 1]`.

### Point time series

```python
with pygndc.open("satellite.gndc") as dataset:
    series = dataset.series(rowcol=(140, 200))
    selected = dataset.series(latlng=(116.5, 39.9), bands=[0, 3])
```

Windowed reads construct coordinates only for the requested area, so memory usage
scales with the requested window rather than the full raster dimensions.

## 6. Analysis Helpers

Vegetation-index methods accept optional `window` or `bbox` selectors:

```python
with pygndc.open("satellite.gndc") as dataset:
    ndvi = dataset.ndvi(t=0, red_band=2, nir_band=3)
    evi = dataset.evi(t=0, blue_band=0, red_band=2, nir_band=3)
    ndwi = dataset.ndwi(t=0, green_band=1, nir_band=3)
    ndmi = dataset.ndmi(t=0, nir_band=3, swir_band=4)
```

Spatial gradients are computed from reconstructed arrays:

```python
dx, dy = dataset.gradient(t=0, mode="spatial")
magnitude = dataset.gradient(t=0, mode="mag")
```

Standalone NumPy helpers are also available:

```python
from pygndc import compute_ndvi, compute_evi, compute_ndwi, compute_ndmi

ndvi = compute_ndvi(red, nir)
```

## 7. Export

```python
with pygndc.open("satellite.gndc") as dataset:
    array = dataset.to_numpy(t=0, window=(0, 0, 512, 512))
    dataset.to_tif("frame.tif", t=0)
    dataset.to_tifs("frames/", name_prefix="MODIS")
```

For xarray integration:

```bash
pip install "pygndc[xarray]"
```

```python
with pygndc.open("satellite.gndc") as dataset:
    xds = dataset.to_xarray(times=[0, 5, 10])
```

## 8. Desktop Viewer

```bash
pip install "pygndc[viewer]"
pygndc viewer
pygndc viewer -i satellite.gndc
```

The viewer supports single- and multi-chunk files and uses native WGPU by default.
Use **Tools > WGPU Adapter** before opening a file to select a particular GPU. Files
created by either native or tcnn encoders are supported. Running `pygndc viewer`
opens an empty viewer; `-i` optionally opens a container after startup.

## 9. Command-Line Examples

```bash
# Metadata
pygndc info satellite.gndc

# Reconstruct stored or interpolated frames
pygndc decompress -i satellite.gndc -o frames/
pygndc decompress -i satellite.gndc -o frame.tif --timestamp 2024-06-15
pygndc decompress -i satellite.gndc -o frames/ \
  --start 2024-01-01 --end 2024-12-31 --interval 5

# Select a decoder
pygndc decompress -i satellite.gndc -o frames/ --mode native_cpu

# Analysis and point extraction
pygndc ndvi -i satellite.gndc -o ndvi.tif --red-band 2 --nir-band 3
pygndc sample -i satellite.gndc --lon 116.5 --lat 39.9
pygndc timeseries -i satellite.gndc --row 140 --col 200 -o series.csv
```

## 10. Low-Level Reader

`GNDCReader` loads one chunk directly. Prefer `pygndc.open()` for multi-chunk files.

```python
from pygndc import GNDCReader

reader = GNDCReader(mode="native_wgpu")
reader.load("single_chunk.gndc")

frame = reader.reconstruct_single_frame(t_idx=0)
window = reader.reconstruct_window(
    t_idx=0,
    row_off=100,
    col_off=100,
    height=512,
    width=512,
)
series = reader.reconstruct_pixel_series(row=140, col=200)
reader.reset()
```

## 11. Container Layout

A v0.9 `.gndc` file is a ZIP STORE container with dataset metadata, a chunk manifest,
one or more neural chunk models, and optional sidecars.

| Entry             | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| `meta.json`       | Shape, axes, CRS, transform, bands, and time metadata      |
| `manifest.json`   | Chunk bounds and file names                                |
| `chunks/*.gndcc`  | Quantized model weights, mask, and optional residual track |
| `preview.png`     | Optional low-resolution overview                           |
| `metrics.json`    | Optional encoding-quality metrics                          |
| `provenance.json` | Optional provenance record                                 |

Missing spatial chunks reconstruct as NoData. Individual chunks may use different
model sizes while remaining accessible through one dataset-level API.
