# pygndc API Reference

This reference describes the public dataset, reader, encoder, backend-selection, and
analysis APIs in pygndc 1.0.11.

## 1. Package-Level Functions

### `pygndc.open()`

```python
pygndc.open(path: str, mode: str = "native_wgpu") -> GNDCDataset
```

Opens a v0.9 `.gndc` container. This is the recommended read API for both single- and
multi-chunk files.

| Parameter | Type  | Default         | Description                                          |
| --------- | ----- | --------------- | ---------------------------------------------------- |
| `path`    | `str` | required        | Container path                                       |
| `mode`    | `str` | `"native_wgpu"` | `native_wgpu`, `native_cpu`, or optional `tcnn_cuda` |

```python
import pygndc

with pygndc.open("data.gndc") as dataset:
    frame = dataset.read(time=0)
```



### `pygndc.list_wgpu_adapters()`

```python
pygndc.list_wgpu_adapters() -> list[dict]
```

Returns WGPU adapters visible to the native runtime. Each record includes `index`,
`name`, `backend`, `device_type`, storage-buffer limits, and `preferred`. Physical
adapters with the same model name are returned as separate records.

### `pygndc.set_wgpu_adapter()`

```python
pygndc.set_wgpu_adapter(name: Optional[str]) -> None
```

Sets the process-wide adapter preference for future WGPU readers and trainers. `name`
is a case-insensitive substring such as `"NVIDIA"`, `"RTX 4080"`, or
`"Vulkan: NVIDIA"`. Pass `None` or an empty string to restore default selection.
Call this function before the first WGPU device is created.

### `pygndc.get_wgpu_adapter()`

```python
pygndc.get_wgpu_adapter() -> Optional[str]
```

Returns the current process-wide WGPU adapter preference.

### `pygndc.set_wgpu_adapter_index()`

```python
pygndc.set_wgpu_adapter_index(index: Optional[int]) -> None
```

Selects one physical adapter using an `index` returned by
`pygndc.list_wgpu_adapters()`. This is the recommended API on multi-GPU systems
with several identical cards. Pass `None` to restore default selection. The
equivalent environment variable is `PYGNDC_WGPU_ADAPTER_INDEX`.

### `pygndc.get_wgpu_adapter_index()`

```python
pygndc.get_wgpu_adapter_index() -> Optional[int]
```

Returns the current process-wide adapter-index preference.

## 2. Inference Modes

| Mode          | Backend              | Device                     | Requirements                                          |
| ------------- | -------------------- | -------------------------- | ----------------------------------------------------- |
| `native_wgpu` | Rust/WGPU            | Vulkan, DX12, or Metal GPU | Standard wheel; runtime CPU fallback                  |
| `native_cpu`  | Rust                 | CPU                        | Standard wheel                                        |
| `tcnn_cuda`   | PyTorch/tiny-cuda-nn | NVIDIA CUDA GPU            | User-installed PyTorch CUDA and tiny-cuda-nn bindings |

`native_wgpu` prefers a discrete adapter. If WGPU initialization or execution fails,
the reader or trainer falls back to native CPU and records the fallback reason.

## 2.1 NPY Training Input

```bash
pygndc tif2npy INPUT_TIF_FOLDER OUTPUT_NPY_FOLDER [--overwrite]
```

Converts a dated GeoTIFF series to the GeoNDC NPY input layout. Each frame is stored
as an HWC `.npy` array with its original dtype. The directory-level `metadata.json`
stores the authoritative frame order, timestamps, CRS, affine transform, nodata value,
band names, shape, and dtype. Conversion does not apply `data_scale`.

Use the prepared directory with the configuration-driven encoder:

```yaml
paths:
  input_path: prepared_npy
data:
  dataset_mode: npy
```

NPY inputs support the same `xy_tile_size`, `t_tile_size`, overlap, block, Native, and
TCNN training paths as standard TIF inputs.

The encoder accepts `train_loss_type: MSE_RRMSE` for band-wise relative MSE. Native and
TCNN derive fixed mean-one output weights from the complete chunk's per-band target energy.
This objective balances per-band RRMSE and is
useful when low-energy bands would otherwise be underweighted by ordinary MSE.

## 3. `GNDCDataset`

```python
GNDCDataset(path: str, mode: str = "native_wgpu")
```

High-level, chunk-aware dataset interface returned by `pygndc.open()`. It supports the
context manager protocol.

### Properties

| Property           | Type                         | Description                                  |
| ------------------ | ---------------------------- | -------------------------------------------- |
| `mode`             | `str`                        | Requested inference mode                     |
| `shape`            | `tuple[int, int, int, int]`  | `(T, H, W, C)`                               |
| `n_timestamps`     | `int`                        | Number of stored observations                |
| `n_bands`          | `int`                        | Number of output bands                       |
| `height`, `width`  | `int`                        | Raster dimensions                            |
| `timestamps`       | `list[datetime]`             | Stored observation times                     |
| `crs`              | `rasterio.crs.CRS` or `None` | Coordinate reference system                  |
| `transform`        | `rasterio.Affine` or `None`  | Pixel-to-CRS transform                       |
| `bounds`           | `BoundingBox`                | Dataset bounds                               |
| `resolution`       | `tuple[float, float]`        | Pixel resolution                             |
| `band_names`       | `Optional[list[str]]`        | Band names                                   |
| `band_wavelengths` | `Optional[list[float]]`      | Center wavelengths                           |
| `nodata`           | `Optional[float]`            | Missing-data fill value                      |
| `has_mask`         | `bool`                       | Whether any chunk stores a mask              |
| `has_residuals`    | `bool`                       | Whether any chunk stores residual correction |
| `meta`             | `dict`                       | Complete dataset metadata                    |
| `manifest`         | `Manifest`                   | Parsed chunk manifest                        |
| `container`        | `Container`                  | Underlying container object                  |

### `metrics()`

```python
dataset.metrics() -> Optional[dict]
```

Returns physical-unit quality metrics recorded at encode time. When present, the
result contains `unit`, `source`, and per-band `rmse`, `mae`, `max`, `n_valid`, and
possibly `psnr`.

### `read()`

```python
dataset.read(
    time: Union[int, str, datetime] = 0,
    *,
    window=None,
    bbox=None,
    rowcol=None,
    latlng=None,
    bands: Optional[list[int]] = None,
) -> numpy.ndarray
```

Reads a stored observation. A string or `datetime` must match a stored timestamp.
Exactly one spatial selector may be supplied.

| Selector | Format                              | Return shape         |
| -------- | ----------------------------------- | -------------------- |
| omitted  | full frame                          | `(H, W, C)`          |
| `window` | `(col_off, row_off, width, height)` | `(height, width, C)` |
| `bbox`   | `(min_x, min_y, max_x, max_y)`      | `(height, width, C)` |
| `rowcol` | `(row, col)`                        | `(C,)`               |
| `latlng` | `(longitude, latitude)`             | `(C,)`               |

```python
frame = dataset.read(time=0)
patch = dataset.read(time=0, window=(100, 200, 512, 512))
pixel = dataset.read(time=0, rowcol=(140, 200), bands=[0, 3])
```

### `interpolate()`

```python
dataset.interpolate(
    time: Union[int, float, str, datetime],
    *,
    window=None,
    bbox=None,
    rowcol=None,
    latlng=None,
    bands: Optional[list[int]] = None,
) -> numpy.ndarray
```

Reconstructs at continuous model time. A float is interpreted as normalized time in
`[0, 1]`; strings and datetimes may fall between stored observations. Spatial selector
and return-shape rules match `read()`.

### `series()`

```python
dataset.series(
    *,
    rowcol=None,
    latlng=None,
    bands: Optional[list[int]] = None,
) -> numpy.ndarray
```

Returns all stored observations at one point with shape `(T, C)`. Exactly one of
`rowcol` and `latlng` is required.

### Vegetation indices

```python
dataset.ndvi(t=0, *, red_band=0, nir_band=1, window=None, bbox=None)
dataset.evi(t=0, *, blue_band=0, red_band=1, nir_band=2, window=None, bbox=None)
dataset.ndwi(t=0, *, green_band=0, nir_band=1, window=None, bbox=None)
dataset.ndmi(t=0, *, nir_band=0, swir_band=1, window=None, bbox=None)
```

Each method returns a float32 array for the requested area.

### `gradient()`

```python
dataset.gradient(
    t: Union[int, str, datetime] = 0,
    *,
    mode: str = "spatial",
    window=None,
    bbox=None,
)
```

Computes finite-difference gradients from reconstructed arrays. `mode="spatial"`
returns `(dx, dy)`; `"dx"`, `"dy"`, `"mag"`, `"spatial_mag"`, or `"temporal"`
returns one array.

### Export

```python
dataset.to_numpy(t=0, *, window=None, bbox=None, bands=None) -> numpy.ndarray
dataset.to_tif(path: str, t=0, *, out_dtype: str = "float32") -> None
dataset.to_tifs(
    output_dir: str,
    *,
    out_dtype: str = "float32",
    progress: bool = True,
    name_prefix: str = "Frame",
) -> None
dataset.to_xarray(times: Optional[list[int]] = None) -> xarray.Dataset
```

`to_xarray()` requires the optional xarray dependency.

### Lifecycle and summary

```python
dataset.info() -> str
dataset.close() -> None
```

`close()` releases chunk readers, GPU resources, and the underlying container.

## 4. `GNDCReader`

```python
from pygndc import GNDCReader

reader = GNDCReader(mode="native_wgpu")
```

The low-level reader loads one chunk directly. For multi-chunk containers, use
`pygndc.open()` unless direct access to a chunk model is required.

### Attributes

| Attribute       | Description                   |
| --------------- | ----------------------------- |
| `mode`          | Requested mode                |
| `backend`       | `native` or `tcnn`            |
| `device`        | `wgpu`, `cpu`, or `cuda`      |
| `model`         | Loaded decoder implementation |
| `model_config`  | Chunk model configuration     |
| `dataset_meta`  | Chunk metadata                |
| `has_residuals` | Residual-track availability   |
| `quant_mode`    | Weight quantization mode      |

### Loading and reset

```python
reader.load(ndc_path: str) -> None
reader.load_chunk_bytes(chunk_bytes, dataset_meta, *, _prefetched_sections=None) -> None
reader.reset() -> None
```

`load()` loads the first chunk of a container. `load_chunk_bytes()` is used by the
dataset router and advanced applications.

### Reconstruction

```python
reader.reconstruct_single_frame(
    t_idx,
    band_idx=-1,
    chunk_size=2**18,
    apply_mask=True,
    add_residuals=True,
) -> numpy.ndarray

reader.reconstruct_window(
    t_idx=None,
    row_off=0,
    col_off=0,
    height=0,
    width=0,
    band_idx=-1,
    chunk_size=2**18,
    apply_mask=True,
    add_residuals=True,
    t_norm=None,
) -> numpy.ndarray

reader.reconstruct_pixel_series(
    row: int,
    col: int,
    chunk_size: int = 2**16,
    apply_mask: bool = True,
    add_residuals: bool = True,
) -> numpy.ndarray
```

`reconstruct_window()` accepts either a stored `t_idx` or continuous normalized
`t_norm`. A single-band request returns `(H, W)`; all bands return `(H, W, C)`.

### Mask and residual access

```python
reader.get_mask(t_idx: int) -> Optional[numpy.ndarray]
reader.get_mask_decimated(t_idx, row_indices, col_indices) -> numpy.ndarray
reader.get_pixel_mask_series(row: int, col: int) -> numpy.ndarray
reader.get_residual(t_idx: int) -> Optional[numpy.ndarray]
```

Mask arrays use `True` for masked/NoData pixels.

## 5. Encoder API

Encoder operations require a valid encoder license. Reader operations do not.
For an evaluation or commercial encoder license, contact `jianboqi@126.com`.

```python
from pygndc.encoder import (
    autocompress,
    estimate_config,
    compress,
    GNDCConfig,
    GNDCCompressor,
)
```

### `autocompress()`

```python
autocompress(
    input_path,
    output_path=None,
    *,
    gpu_mem_gb=None,
    dry_run=False,
    xy_tile_size=None,
    t_tile_size=None,
    block_size=None,
)
```

Probes an input, estimates model and chunk parameters, and writes a `.gndc` container.
When `dry_run=True`, prints the plan, skips training, and returns `None`.

### `estimate_config()`

```python
estimate_config(
    input_path,
    output_path=None,
    *,
    gpu_mem_gb=None,
    dataset_mode=None,
    layout=None,
    xy_tile_size=None,
    t_tile_size=None,
    block_size=None,
) -> GNDCConfig
```

Returns an editable configuration without training. `dataset_mode` and `layout` are
used for compound products such as dual-variable multiband archives.

### `compress()`

```python
compress(config: Union[GNDCConfig, str, Path]) -> str
```

Runs encoding strictly from a `GNDCConfig` or YAML file and returns the output path.

### `GNDCConfig`

```python
GNDCConfig(config_path=None)
GNDCConfig.from_yaml(path) -> GNDCConfig
GNDCConfig.from_dict(values: dict) -> GNDCConfig
```

Important fields include:

| Group    | Fields                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------- |
| Input    | `input_folder`, `dataset_mode`, `layout`, `data_scale`, `input_valid_range`                    |
| Backend  | `train_mode`, `train_dtype`, `cords_norm_mode`                                                 |
| Training | `epochs`, `batch_size`, `lr`, `lr_hash`, `train_loss_type`                                     |
| Model    | `xy_*`, `t_*`, `decoder_hidden`, `n_hidden_layers`, `disable_temporal`                         |
| Sampling | `importance_sampling`, `imp_alpha`, `imp_warmup_frac`, `imp_refresh_epochs`                    |
| Chunking | `tiled_enabled`, `xy_tile_size`, `t_tile_size`, `tile_overlap`, `t_tile_overlap`, `block_size` |
| Output   | `output_model`, `quantization_bit`, `mask_mode`, `save_residuals`, `residual_threshold`        |

`train_mode` accepts `native_wgpu`, `native_cpu`, and `tcnn_cuda`. New configurations
default to `native_wgpu`, float32 native training, MSE, and `cords_norm_mode=3`.

### `GNDCCompressor`

```python
GNDCCompressor(
    train_dtype="auto",
    global_start_date=None,
    global_end_date=None,
    train_mode="native_wgpu",
)
```

Low-level encoder for applications that manage loading and training stages directly.
When `train_mode="tcnn_cuda"`, construction dispatches to the tcnn implementation.

Primary methods:

```python
compressor.load_dataset_folder(...)
compressor.load_dataset_zarr(...)
compressor.load_dataset_zarr_streaming(...)
compressor.prepare_training_data_optimized(
    use_log=False,
    norm_mode=3,
    force_zero_training=False,
    block_size=(1, 1),
)
compressor.build_gndc_model(...)
compressor.fit(
    epochs=100,
    batch_size=2**19,
    lr=1e-2,
    lr_hash=None,
    loss_type="mse",
    importance_sampling=False,
    ...,
)
compressor.compute_residuals(threshold=0.005, batch_size=2**20)
compressor.compute_quality_metrics(n_frames=24, batch_size=2**20)
compressor.save_gndc(...)
compressor.save_chunk_blob(...)
```

Native training supports MSE and one Adam learning rate. `lr_hash` is accepted for
configuration compatibility and is normalized to `lr`. tcnn additionally supports
separate hash/MLP learning rates and optional alternative losses.

## 6. Licensing API

```python
from pygndc.licensing import (
    fingerprint,
    request_payload,
    install_license,
    license_status,
    require_encoder_license,
)
```

| Function                                    | Description                                                  |
| ------------------------------------------- | ------------------------------------------------------------ |
| `fingerprint()`                             | Returns the current machine identifier                       |
| `request_payload()`                         | Builds the license-request dictionary                        |
| `install_license(source, destination=None)` | Installs a signed license file                               |
| `license_status()`                          | Returns a `LicenseStatus` record                             |
| `require_encoder_license()`                 | Raises `RuntimeError` unless encoder authorization is usable |

The native extension performs the authoritative signature, machine, feature, and
expiry checks for encoder entry points.

## 7. Standalone Analysis Functions

```python
from pygndc import (
    compute_ndvi,
    compute_evi,
    compute_ndwi,
    compute_ndmi,
    compute_temporal_gradient,
    compute_spatial_gradient,
)
```

| Function                                                  | Operation                            |
| --------------------------------------------------------- | ------------------------------------ |
| `compute_ndvi(red, nir)`                                  | `(nir - red) / (nir + red)`          |
| `compute_evi(blue, green, nir, g=2.5, c1=6, c2=7.5, l=1)` | Enhanced Vegetation Index            |
| `compute_ndwi(green, nir)`                                | Normalized Difference Water Index    |
| `compute_ndmi(nir, swir)`                                 | Normalized Difference Moisture Index |
| `compute_temporal_gradient(data)`                         | NumPy temporal gradient              |
| `compute_spatial_gradient(data)`                          | NumPy spatial gradients `(dx, dy)`   |

These functions operate on NumPy arrays and are independent of the neural decoder.

## 8. Exceptions

All package-specific exceptions inherit from `GNDCError`:

```text
ConfigError
DataLoadError
ModelError
CompressionError
DecompressionError
FileFormatError
DeviceError
CoordinateError
```

## 9. Viewer Command

```bash
pygndc viewer [--mode native_wgpu|native_cpu|tcnn_cuda] [-i FILE.gndc]
```

Both `--mode` and `-i` are optional. Without `-i`, the command opens an empty viewer
and the user can select a container from **File > Open**. On Windows, the command
enables per-monitor DPI awareness before creating the Tk window.
