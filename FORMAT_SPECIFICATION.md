# GNDC Format Specification — v0.9

**GNDC = Geographic Neural Data Cube.**

A file format for N-dimensional geospatial-temporal data cubes
(typically satellite-imagery time series) compressed via neural-field
representations. A single `.gndc` file packages one such cube as one or
more trained models, each covering a contiguous N-dimensional region
("chunk") of the cube. Reading is a forward pass on continuous
coordinates inside a chunk; chunks may be sparsely populated (some
regions can be skipped — they return NaN) and may be split along any
axis (spatial, temporal, or both).

**Status:** Pre-release. Format is frozen at this version for v0.9.x point
releases; breaking changes go to v1.0.

**Scope:** Defines the byte-level layout of `.gndc` files (the user-facing
container) and `.gndcc` files (chunks inside the container). Goal: a
third-party reader can be implemented end-to-end from this document.

**Relationship to existing standards:** GNDC is data-storage-only and
does not duplicate (a) data discovery / catalog functionality — use
[STAC](https://stacspec.org/) and treat a `.gndc` as a STAC `Asset`;
(b) coordinate-reference-system semantics — use WKT2 (ISO 19162:2019)
or EPSG codes parsed by PROJ / pyproj. Metadata field names are
aligned with [CF Conventions](https://cfconventions.org/) where
practical (`scale_factor`, `add_offset`, `valid_range`, `_FillValue`).

---

## 1. Conceptual model

A `.gndc` file is a self-contained published data product representing
a Geographic Neural Data Cube — an N-dimensional cube (typically
`(time, y, x, band)`) compressed as one or more trained neural networks.

```
┌─ data cube (T, H, W, C) ──────────────────┐
│                                            │
│   ┌─────────┬─────────┬─────────┬────────┐│
│   │chunk    │chunk    │chunk    │chunk   ││  one chunk per
│   │(0,0,0)  │(0,0,1)  │(0,0,2)  │(0,0,3) ││  (t_idx, y_idx, x_idx)
│   ├─────────┼─────────┼─────────┼────────┤│
│   │(0,1,0)  │(0,1,1)  │   ...   │  ...   ││  cell of the grid
│   └─────────┴─────────┴─────────┴────────┘│
│   ... and another time slice at t=1 ...    │
└────────────────────────────────────────────┘
```

A **chunk** = one trained model + its mask, covering a contiguous N-D
region. Chunks need not all exist; missing regions return NaN (or a
user-configured fill) at read time. The model inside a chunk handles
forward inference on arbitrary continuous coordinates within that
chunk's bounds — GNDC is natively resolution-free, so there is no
pyramid / overview concept.

A container with exactly one chunk is the simplest case. Containers
with N×M×… chunks support large images split spatially, long time series
split temporally, or both simultaneously.

**Chunk-boundary discontinuity & the `overlap` halo.** Each chunk's model
is trained independently, so at chunk boundaries the reconstructed values
may show seam-like discontinuities of magnitude comparable to the chunk's
training residual (typically &lt; 1 LSB at the published quantization, but
visually noticeable on extreme-stretch displays). v0.9 mitigates this with
an optional **overlap halo**: when `manifest.overlap` (px) is &gt; 0, each
chunk is trained on its `bounds` region **plus** an `overlap`-wide margin on
every spatial side (clamped to the image), so its stored model can produce
valid pixels over the extended extent `clamp(bounds ± overlap, [0, dim])`.
A reader that supports overlap **feather-blends** the shared band — weight
ramps linearly from 1 inside `bounds` to ~0 at the stored edge, normalized
across contributors — making the transition continuous and the seam vanish.
`overlap` is additive and backward-compatible: absent or `0` means
legacy edge-to-edge tiling (each pixel owned by exactly one chunk), and a
reader that ignores the field still produces correct (if seamed) output,
since `bounds` always denote the core, non-overlapping region.

---

## 2. File extensions

| Extension | Meaning                              | User-facing? |
|-----------|--------------------------------------|--------------|
| `.gndc`   | Container (ZIP STORE archive)        | Yes          |
| `.gndcc`  | Chunk binary blob (inside container) | No (internal) |

The container is what users download, share, and open with `pygndc.open()`.
The chunk binary is an implementation detail visible only when you
`unpack` the container.

---

## 3. Container layout

A `.gndc` container is a **ZIP STORE** archive (no compression at the
ZIP layer — chunk weights are already quantized + zstd-compressed
inside each chunk file). `STORE` keeps inner chunk bytes byte-identical
whether read from inside the container or after `unpack`, and the ZIP
central directory at end-of-file enables O(1) random access by name
(including HTTP range requests).

**Required entries:**

```
container.gndc
├── meta.json          REQUIRED  — dataset-level metadata
├── manifest.json      REQUIRED  — chunk index
└── chunks/            REQUIRED  — directory of chunk binaries
    ├── t0000_y0000_x0000.gndcc
    ├── t0000_y0000_x0001.gndcc
    └── ...
```

**Optional entries (sidecars):**

```
├── preview.png        OPTIONAL  — first-paint thumbnail
├── metrics.json       OPTIONAL  — per-chunk quality metrics
├── provenance.json    OPTIONAL  — training params + source provenance
├── thumbnails/        OPTIONAL  — per-band or per-time thumbnails
│   └── ...
└── ...                            other vendor extensions allowed
```

A reader MUST ignore any container entry it does not recognize. A reader
MAY emit a warning for unrecognized top-level entries.

ZIP comment / extra-field data is unused by this format and SHOULD NOT
be relied upon.

---

## 4. `meta.json`

UTF-8 encoded JSON document at the root of the container.

```json
{
  "format": "gndc",
  "version": "0.9",
  "shape": [438, 5401, 5401, 2],
  "axes": ["time", "y", "x", "band"],
  "dtype": "uint8",

  "bands": [
    {
      "name": "LAI",
      "description": "Leaf Area Index",
      "unit": "m^2/m^2",
      "scale_factor": 0.1,
      "add_offset": 0.0,
      "valid_range": [0.0, 10.0],
      "fill_value": 255,
      "center_wavelength": null,
      "common_name": null
    },
    {
      "name": "FPAR",
      "description": "Fraction of Absorbed PAR",
      "unit": "1",
      "scale_factor": 0.004,
      "add_offset": 0.0,
      "valid_range": [0.0, 1.0],
      "fill_value": 255
    }
  ],

  "missing_fill": null,
  "use_log": false,

  "crs": "EPSG:4326",
  "transform": {
    "a": 0.008333, "b": 0.0, "c": 117.0,
    "d": 0.0, "e": -0.008333, "f": 42.0
  },

  "time": {
    "values": [
      "2018-01-01T00:00:00Z",
      "2018-01-06T00:00:00Z"
    ],
    "calendar": "gregorian",
    "global_start": "2018-01-01T00:00:00Z",
    "global_end":   "2024-09-26T00:00:00Z"
  }
}
```

### 4.1 Required fields

| Field      | Type               | Description |
|------------|--------------------|-------------|
| `format`   | string             | MUST be `"gndc"`. |
| `version`  | string             | MUST be `"0.9"` (or compatible 0.9.x). |
| `shape`    | array of int       | Full data cube shape; length must equal `axes` length. |
| `axes`     | array of string    | Axis names in `shape` order. Reserved: `"time"`, `"y"`, `"x"`, `"band"`. Vendors MAY define custom axes (must be documented per-product). |
| `dtype`    | string             | Source data dtype before scaling: `"uint8"`, `"uint16"`, `"int16"`, `"float32"`, `"float16"`. Reader returns `float32` physical units after `scale_factor + add_offset`. |

### 4.2 Bands (recommended)

`bands` is the **single source of truth** for per-band metadata, replacing
the older parallel arrays. CF-aligned field names:

| Field               | Type                | Description |
|---------------------|---------------------|-------------|
| `name`              | string  (required)  | Short identifier, e.g. `"LAI"`. |
| `description`       | string  (opt)       | Human-readable description. |
| `unit`              | string  (opt)       | [UDUNITS-2](https://docs.unidata.ucar.edu/udunits/current/) string. `"1"` for dimensionless. |
| `scale_factor`      | number  (default 1.0) | Multiplier in `physical = stored * scale_factor + add_offset` (CF). |
| `add_offset`        | number  (default 0.0) | Offset in the same formula. |
| `valid_range`       | [min, max]          | Valid range in PHYSICAL units. |
| `fill_value`        | number  (opt)       | Stored value that means nodata (CF `_FillValue`). Distinct from `missing_fill`. |
| `center_wavelength` | number  (opt)       | nm, for optical sensors (STAC `eo:bands.center_wavelength`). |
| `common_name`       | string  (opt)       | STAC `eo:bands.common_name`: `"red"`, `"nir"`, etc. |

Length of `bands` MUST equal `shape[axes.index("band")]`.

### 4.3 Recommended geospatial fields

| Field            | Type                | Description |
|------------------|---------------------|-------------|
| `crs`            | string              | Coordinate reference system: prefer **WKT2** (ISO 19162:2019), accept WKT1 or `"EPSG:NNNN"`. Readers SHOULD parse via PROJ / pyproj. May be `null` for non-geo data. |
| `transform`      | object              | Affine transform (see §4.4). May be `null` if no geo. |
| `missing_fill`   | number or null      | Value the reader inserts for regions covered by **no chunk** (i.e. cells absent from the manifest). `null` ⇒ NaN (default). Distinct from each band's `fill_value`. |
| `use_log`        | bool                | Whether training applied log-transform; reader applies inverse. Default `false`. |

### 4.4 Affine transform

`transform` is an object with explicit named coefficients to remove the
"GDAL vs rasterio order" ambiguity. The mapping from pixel (col, row)
to world (x, y) is:

```
x = a*col + b*row + c
y = d*col + e*row + f
```

| Key | Meaning (north-up images)           |
|-----|-------------------------------------|
| `a` | pixel width along x                 |
| `b` | row rotation (0 for north-up)       |
| `c` | x of upper-left pixel center *or* corner — pick one convention, document in `pixel_anchor` below |
| `d` | column rotation (0 for north-up)    |
| `e` | pixel height along y (typically negative) |
| `f` | y of upper-left pixel reference     |

Optional companion field:

| Field           | Type   | Description |
|-----------------|--------|-------------|
| `pixel_anchor`  | string | `"upper_left"` (default; pixel reference is the UL corner of the cell, GDAL convention) or `"center"` (cell-center, CF convention). |

This object matches `rasterio.Affine(a, b, c, d, e, f)`. GDAL
`GeoTransform()` returns the same coefficients in a different order:
`(c, a, b, f, d, e)` — readers / writers using GDAL must reorder.

### 4.5 Time (recommended)

The `time` object groups all temporal metadata.

| Field           | Type             | Description |
|-----------------|------------------|-------------|
| `values`        | array of string  | One per time index. ISO 8601, MUST include timezone (`Z` or `±HH:MM`). Length == `shape[axes.index("time")]`. |
| `calendar`      | string           | CF calendar name (default `"gregorian"`). Allowed: `gregorian`, `proleptic_gregorian`, `noleap`, `360_day`, `julian`. |
| `global_start`  | string           | ISO 8601 with timezone. Used by readers for time-normalization. |
| `global_end`    | string           | ISO 8601 with timezone. |
| `units`         | string  (opt)    | CF-style `units = "days since 2018-01-01T00:00:00Z"` for regular cadence. Mutually exclusive with `values`. |

Either `values` OR (`units` + `global_start` + `global_end`) MUST be
present if the cube has a `time` axis. All timestamps are interpreted
in UTC unless an explicit offset is present.

### 4.6 Forward compatibility

A reader MUST ignore unknown fields in `meta.json`. A writer MAY add
vendor-specific fields prefixed with `x_` to avoid collisions.

---

## 5. `manifest.json`

The chunk index. Reader uses this and only this to find chunks.

```json
{
  "format": "gndc",
  "version": "0.9",
  "chunk_grid": {
    "time": {"n": 1, "regular": true, "size": 438},
    "y":    {"n": 4, "regular": true, "size": 1351},
    "x":    {"n": 4, "regular": true, "size": 1351}
  },
  "overlap": 0,
  "chunks": [
    {
      "id": [0, 0, 0],
      "bounds": {
        "time": [0, 438],
        "y":    [0, 1351],
        "x":    [0, 1351]
      },
      "file": "chunks/t0000_y0000_x0000.gndcc",
      "size_bytes": 14782345
    },
    {
      "id": [0, 0, 1],
      "bounds": {
        "time": [0, 438],
        "y":    [0, 1351],
        "x":    [1351, 2702]
      },
      "file": "chunks/t0000_y0000_x0001.gndcc",
      "size_bytes": 15123821
    }
  ],
  "missing": []
}
```

### `chunk_grid` (informational)

A compact summary of regular chunking. If chunks are irregular, set
`regular: false` and `size` may be omitted.

For each axis:
| Field    | Type | Description |
|----------|------|-------------|
| `n`      | int  | Number of chunks along this axis. |
| `regular`| bool | True if all chunks along this axis have the same size (except possibly the last). |
| `size`   | int  | Chunk size along this axis (only if `regular: true`). |

`chunk_grid` is informational. The authoritative chunk list is `chunks`;
a reader MUST NOT infer chunk existence from `chunk_grid` alone.

### `overlap` (optional, top-level)

| Field     | Type | Description |
|-----------|------|-------------|
| `overlap` | int  | Spatial halo width (px) each chunk was trained with beyond its core `bounds`. Default `0`. |

When `overlap > 0`, every chunk's stored model is valid over
`clamp(bounds ± overlap, [0, dim])` on the spatial axes; readers SHOULD
feather-blend the shared band to hide seams (see §2). `bounds` always
remain the core, non-overlapping region, so the field is purely additive —
omit it (or set `0`) for legacy edge-to-edge tiling.

### `chunks` (authoritative)

One entry per **existing** chunk. Missing chunks are simply absent from
this list.

| Field         | Type            | Description |
|---------------|-----------------|-------------|
| `id`          | array of int    | Chunk identifier in axis order: `[t_idx, y_idx, x_idx, ...]`. |
| `bounds`      | object          | Half-open `[start, end)` ranges per axis (in **global** indices). |
| `file`        | string          | Path inside the ZIP to the `.gndcc` chunk binary. |
| `size_bytes`  | int             | Uncompressed size of the chunk binary (informational). |
| `metrics`     | object (opt)    | Per-chunk quality metrics, e.g. `{"psnr": 39.3, "rmse": 0.07}`. |

`bounds` keys MUST match `meta.json` `axes` for axes that participate in
chunking; axes the chunk spans entirely MAY be omitted.

### `missing` (optional, informational)

```json
"missing": [
  {"id": [0, 2, 1], "reason": "ocean_only", "tag": "skip"},
  {"id": [1, 0, 0], "reason": "train_oom", "tag": "error"}
]
```

Purely diagnostic. Readers MAY use it to surface "why" to the user, but
MUST NOT treat absence of an entry from `missing` as proof a chunk
exists — only `chunks` is authoritative.

### 5.4 Per-chunk affine transform

A chunk does not store its own affine transform. The reader derives
the chunk-local transform from the outer `meta.transform` plus the
chunk's `bounds`:

```
chunk_transform = {
  a: meta.transform.a,
  b: meta.transform.b,
  c: meta.transform.c + meta.transform.a * chunk.bounds.x[0]
                      + meta.transform.b * chunk.bounds.y[0],
  d: meta.transform.d,
  e: meta.transform.e,
  f: meta.transform.f + meta.transform.d * chunk.bounds.x[0]
                      + meta.transform.e * chunk.bounds.y[0]
}
```

(For north-up images, `b = d = 0` and the formula simplifies to a
translation of `c` by `a*x_start` and `f` by `e*y_start`.) This keeps
the manifest the single source of geospatial truth.

---

## 6. Chunk binary format (`.gndcc`)

Each chunk is a self-contained binary blob holding the trained model
weights, per-frame mask, and per-chunk training-time metadata. Outer
container metadata (timestamps, CRS, etc.) is NOT duplicated here.

### Byte layout

```
Offset       Size    Description
─────────    ────    ────────────────────────────────────────────────
  0          4       chunk_meta_len      uint32 little-endian
  4          N       chunk_meta_json     UTF-8 JSON, length N == chunk_meta_len
  4+N        4       weights_len         uint32 little-endian
  8+N        W       weights_bytes       zstd-compressed quantized weights
  8+N+W      4       mask_len            uint32 little-endian
  12+N+W     M       mask_bytes          zstd-compressed mask (see §6.3)
 [optional residuals trailer, present iff chunk_meta.has_residuals]
  12+N+W+M   4       residual_len        uint32 little-endian
  16+N+W+M   R       residual_bytes      sparse residual block (see §6.4)
```

All multi-byte numeric fields in the chunk binary (integers AND floats)
are little-endian.

### 6.1 `chunk_meta_json` schema

```json
{
  "model_config": {
    "xy_hash_size": 20,
    "xy_base_resolution": 64,
    "xy_n_levels": 10,
    "xy_per_level_scale": 1.41,
    "xy_features_per_level": 4,
    "t_hash_size": 20,
    "t_base_resolution": 64,
    "t_n_levels": 10,
    "t_per_level_scale": 1.30,
    "t_features_per_level": 4,
    "spatial_scale": 0.1,
    "decoder_hidden": 64,
    "n_hidden_layers": 3,
    "disable_temporal": false,
    "block_size": [1, 1],
    "n_bands": 2,
    "spatial_dims": 2
  },

  "quantization_info": {
    "mode": "int8_per_tensor_f16dec",
    "scales": [158.5, 181.8],
    "sizes":  [12233344, 36221472],
    "components": ["spatial", "phenology"],
    "decoder_size": 14336,
    "decoder_name": "decoder"
  },

  "train_data_range": [[0.0, 6.6], [0.0, 0.98]],

  "train_time_range": {
    "start": "2018-01-01T00:00:00",
    "end":   "2024-09-26T00:00:00",
    "n_frames": 438
  },

  "has_residuals": true,
  "residual_info": {
    "dtype": "int16",
    "threshold": 0.005,
    "scale": 1.0,
    "main_codec": "zstd",
    "outlier_layout": "chunked_v1",
    "outlier_count": 12345,
    "outlier_value_scale": 0.0123,
    "outlier_vals_codec": "lzma"
  }
}
```

Fields:

| Field              | Required | Description |
|--------------------|----------|-------------|
| `model_config`     | yes      | Hyperparameters used to build the GNDCNetwork. Must contain enough to reinstantiate the exact same architecture. |
| `quantization_info`| yes      | Weight dequantization recipe (see §6.2). |
| `train_data_range` | yes      | Per-band [min, max] used to normalize data during training. Reader uses for denormalization. |
| `train_time_range` | required if model uses temporal branch | `start` + `end` define the time interval the temporal hash grid was trained against. Reader normalizes query times into [0, 1] against this range before invoking the model. |
| `has_residuals`    | yes      | If true, the residual trailer is present. |
| `residual_info`    | required if `has_residuals` | Recipe for decoding the residual trailer (see §6.4). |

### 6.2 `quantization_info` modes

Stored under `quantization_info.mode`:

| Mode                            | Description |
|---------------------------------|-------------|
| `"float16"`                     | Weights stored raw as float16. No scale needed. |
| `"int8_per_tensor_f16dec"`      | Hash-grid components (spatial, [phenology]) quantized to int8 with per-tensor scales; decoder stored at float16. **Default in v0.9.** |
| `"int16_per_tensor_f16dec"`     | Same layout, int16 grids. |
| `"int8_per_tensor"`             | All components quantized to int8 with per-tensor scales (decoder included). |
| `"int16_per_tensor"`            | Same, int16. |

For `*_per_tensor*` modes:

| Field          | Type           | Description |
|----------------|----------------|-------------|
| `scales`       | array of float | One per grid component. Dequant: `raw_float = int_val / scale`. |
| `sizes`        | array of int   | Element count per grid component, in concatenation order. |
| `components`   | array of string| Names of grid components, in order: `["spatial"]` or `["spatial", "phenology"]`. |
| `decoder_size` | int (f16dec only) | Element count of decoder block (stored as float16 after grid block). |
| `decoder_name` | string (opt)   | Diagnostic name (e.g. `"decoder"`). |

Weight payload layout (concatenated):
1. Grid components in `components` order, each `sizes[i]` ints of the
   declared dtype.
2. If `_f16dec`: the decoder, `decoder_size` float16 values.

After dequant + concat, the full weight tensor is loaded into
GNDCNetwork via the model's `load_weights_from_flat_buffer`.

### 6.3 Mask format

Inside `mask_bytes`:

```
Offset  Size   Field
   0    1      mask_type    uint8     0=static, 1=dynamic, 255=none
   1    12     shape        3× uint32 LE — (T, H, W) for dynamic, (1, H, W) for static
  13    ...    per-frame zstd chunks (one per frame of shape[0])
```

Each per-frame chunk:
```
  4      frame_len    uint32 LE
  N      frame_bytes  zstd-compressed bitmap (packed bool, MSB first, row-major)
```

Decoded bitmap is a `(H, W)` `bool` array where `True` = **nodata**
(invalid / masked out). For static mode, the single decoded frame
applies to all times.

If `mask_type == 255` (no mask), the mask section after the
12-byte shape header is empty.

### 6.4 Residual format

When `has_residuals: true`, the residual trailer follows the mask block
inside the chunk binary. Residuals are sparse error-correction values:
a small per-pixel `int16` stream covering the bulk of the volume, plus
a per-frame "outlier track" for values that overflow int16 quantization.

A reader MAY ignore the residual trailer entirely and serve
**unrectified** reconstructions (model output only); the result is still
valid pygndc data, just without the residual correction applied.

#### Residual byte layout

```
Offset within residual_bytes        Field
─────────────────────────────────   ────────────────────────────────────
  0      4   uint32 LE              main_len
  4      M   bytes                  main_compressed              (§6.4.1)
  4+M    1   uint8                  has_outliers   (0 or 1)
  if has_outliers == 1:
       16   int32 LE × 4            shape (T, H, W, C)
        4   uint32 LE               n_offsets   (= T+1)
        (T+1) × 8   uint64 LE       chunk_offsets[i]
        — followed by T per-frame chunks (§6.4.2). chunk_offsets[i]
          gives the byte offset of frame i's chunk relative to the
          first per-frame byte; chunk_offsets[T] equals the total
          chunked-data byte length (sentinel).
```

#### 6.4.1 `main_compressed`

The compressed in-band residual stream. Compression codec is given by
`residual_info.main_codec` (`"zstd"` is the default; `"lzma"` allowed).
After decompression, the bytes form a contiguous `(T, H, W, C)` array
of `residual_info.dtype` (currently always `int16`) in row-major scan
order. Physical residual values:

```
residual_phys = main_int16 * residual_info.scale
```

#### 6.4.2 Per-frame outlier chunks

Each frame's chunk has the layout:

```
  4   float32 LE   frame_scale
  4   uint32 LE    pos_len
  B   bytes        pos_zstd           (zstd of int32 LE gap-deltas; see below)
  4   uint32 LE    vals_count_in_frame
  4   uint32 LE    vals_len
  V   bytes        vals_compressed    (codec = residual_info.outlier_vals_codec)
```

The **position field** (`pos_zstd`) is the zstd of an `int32` little-endian
array of `vals_count_in_frame` **gap-deltas** of the frame's outlier flat
positions (`pos = row*W*C + col*C + band`, scan order, sorted ascending):
`deltas[0] = pos[0]`, `deltas[k] = pos[k] - pos[k-1]`, so
`pos = cumsum(deltas)`. Outliers are sparse and scattered (~0.1% of pixels),
for which delta-coding is ~20–30% smaller than a packed bitmap and decodes
without a full-frame `unpackbits` (no `O(H*W*C)` work per frame).

> Format note: files written before this revision stored a packed-bool
> **bitmap** in the position field instead (same surrounding layout). The two
> are not self-describing — a reader decodes whichever its version expects;
> bitmap files must be transcoded (`scripts/transcode_residual_delta.py`).

Empty frames (no outliers) write a chunk with `pos_len=0`,
`vals_count_in_frame=0`, `vals_len=0`, and `frame_scale=1.0`.

Decoding a frame:

1. zstd-decompress `pos_zstd` → `int32` deltas of length
   `vals_count_in_frame`; outlier positions = `cumsum(deltas)` (scan order).
2. Decompress `vals_compressed` (codec from
   `residual_info.outlier_vals_codec`; default `"lzma"`) → array of
   `int16`/`int8` (per `residual_info.dtype`) of length `vals_count_in_frame`.
3. For each position `p[k]`, assign:
   `outlier_residual_phys[p[k]] = vals[k] * frame_scale` (per-frame mode), or
   `vals[k] * band_scales[p[k] % C]` (native mode; `frame_scale` is a `1.0`
   sentinel).

After this pass, the frame's full residual = in-band residual from
`main_compressed` PLUS the outlier residual at the marked positions.

#### `residual_info` fields recap

| Field                | Type   | Description |
|----------------------|--------|-------------|
| `dtype`              | string | In-band residual dtype. Currently `"int16"`. |
| `threshold`          | float  | Training-time error threshold above which a sample becomes an outlier. Informational. |
| `scale`              | float  | Physical scale for `main_compressed` int16 values. |
| `main_codec`         | string | `"zstd"` (default) or `"lzma"`. |
| `outlier_layout`     | string | `"chunked_v1"`. Fixed in v0.9. |
| `outlier_count`      | int    | Total outliers across all frames (sum of `vals_count_in_frame`). |
| `outlier_value_scale`| float  | Informational global outlier scale; per-frame scales are inline. |
| `outlier_vals_codec` | string | `"lzma"` (default) or `"zstd"`. |

---

## 7. Reader semantics

A conformant reader exposes (at minimum) these operations:

```python
ds = pygndc.open("container.gndc")

ds.shape           # (T, H, W, C) — from meta.json
ds.timestamps      # list[str] from meta.json
ds.crs, ds.transform, ds.bounds
ds.coverage()      # ndarray of bool: which chunks exist
ds.read(time=t, window=(col, row, w, h))   # (h, w, C) float32
ds.read(time=t, rowcol=(r, c))              # (C,) float32
ds.series(rowcol=(r, c))                    # (T, C) float32
```

### Query routing

For `read(time=t, window=(col, row, w, h))`:

1. Translate `t` to global frame index via `meta.timestamps` (string,
   datetime, or int).
2. Find all chunks whose `bounds` intersect `(time=[t,t+1], y=[row, row+h], x=[col, col+w])`.
3. For each matched chunk:
   - Convert global indices to chunk-local indices using
     `chunk.bounds.<axis>[0]`.
   - Open the chunk (load model + weights).
   - Run forward pass to produce the chunk's contribution.
   - Paste into the output buffer at the right offset.
4. Cells NOT covered by any chunk MUST be filled with `meta.fill_value`
   (default NaN).

### Time normalization inside a chunk

The chunk model's temporal hash grid was trained against the chunk's
own `train_time_range`. The reader converts a global frame index `t`
into the chunk-local normalized time:

```
t_local = (datetime(t) - chunk.train_time_range.start) /
          (chunk.train_time_range.end - chunk.train_time_range.start)
```

This is the value passed to the model's t input.

### Missing chunk behavior

By default, regions covered by no chunk are filled with NaN
(or `meta.fill_value` if set). Readers MAY expose a strict mode that
raises on missing-region queries instead.

---

## 8. Encoder responsibilities

A conformant encoder MUST:

1. Write `meta.json` and `manifest.json` as the first two ZIP entries
   (helpful for streaming readers, not strictly required).
2. Ensure every entry listed in `manifest.json.chunks` is present in the
   ZIP under the named path.
3. Ensure no entry is listed twice in `chunks`.
4. Ensure `chunks[*].bounds` use **global** axis indices (not chunk-local).
5. Use ZIP STORE (no compression) for all entries.
6. Use UTF-8 for all JSON.
7. Use little-endian for all binary integers.

An encoder MAY:

- Include any sidecars listed in §3.
- Set `metrics` per chunk if PSNR/RMSE were computed at training time.

---

## 9. Reference values

The first official v0.9 product (HiGLASS LAI+FPAR n042e117) has:

- shape: `(438, 5401, 5401, 2)`
- chunks: 16 (4×4 spatial, single temporal chunk covering full series)
- quantization: `int8_per_tensor_f16dec`
- container size: ~250 MB
- global PSNR: LAI 39.4 dB, FPAR 37.9 dB

---

## 10. Versioning policy

| Version | Compatibility               |
|---------|-----------------------------|
| 0.9.x   | Frozen; bug fixes only.     |
| 1.0     | First stable release. May break 0.9 if needed; will be a deliberate decision. |
| ≥ 1.0   | Backward compatible within major version. |

Readers SHOULD check `version` and reject `version` strings whose major
component they do not understand.

---

## 11. STAC integration

A `.gndc` file is one valid storage backend for a STAC
[Asset](https://github.com/radiantearth/stac-spec/blob/master/item-spec/item-spec.md#asset-object).
GNDC does not embed STAC metadata in the container and does not provide
catalog / discovery functionality — those concerns belong to STAC.

**Recommended publication pattern:**

```
my_product/
├── n042e117.gndc            ← the data
└── n042e117.stac.json       ← STAC Item describing the asset
```

A minimal STAC Item referring to a `.gndc` asset:

```json
{
  "stac_version": "1.0.0",
  "type": "Feature",
  "id": "higlass-n042e117-2018-2024",
  "geometry": {"type": "Polygon", "coordinates": [[...]]},
  "bbox": [117.0, 42.0, 162.0, 87.0],
  "properties": {
    "datetime": null,
    "start_datetime": "2018-01-01T00:00:00Z",
    "end_datetime":   "2024-09-26T00:00:00Z"
  },
  "assets": {
    "data": {
      "href": "./n042e117.gndc",
      "type": "application/vnd.gndc+zip",
      "roles": ["data"],
      "title": "HiGLASS LAI+FPAR, 2018-2024",
      "gndc:version": "0.9",
      "gndc:chunks": 16
    },
    "thumbnail": {
      "href": "./n042e117.gndc#preview.png",
      "type": "image/png",
      "roles": ["thumbnail"]
    }
  }
}
```

Suggested MIME type: `application/vnd.gndc+zip`.

The `gndc:*` properties are GNDC-specific STAC extensions — usable but
not required.

---

## 12. `provenance.json` (optional sidecar)

If present, this sidecar documents how the container was produced. Goal:
audit trail and reproducibility. Minimal recommended schema:

```json
{
  "format": "gndc-provenance",
  "version": "0.9",
  "encoder": {
    "name":    "pygndc",
    "version": "1.0.3",
    "commit":  "f680aaf"
  },
  "created":  "2026-05-31T15:30:00Z",
  "duration_seconds": 2940,

  "source": {
    "uri":      "F:/NDC-Data/NDC_Datasets/HiGLASS_LAI_FPAR/n042e117/",
    "format":   "GeoTIFF LZW (uint8)",
    "n_files":  12,
    "checksum": null
  },

  "training": {
    "loss":       "MSE",
    "epochs":     5,
    "batch_size": 524288,
    "optimizer":  "Adam",
    "lr_grid":    0.005,
    "lr_mlp":     0.01,
    "hardware":   "NVIDIA RTX 4090 (24 GB)",
    "framework":  "pytorch 2.6.0+cu124"
  },

  "quality_summary": {
    "psnr_db":  {"LAI": 39.41, "FPAR": 37.91},
    "rmse":     {"LAI": 0.074, "FPAR": 0.013},
    "rel_max":  {"LAI": 0.011, "FPAR": 0.013}
  }
}
```

All fields are optional except `format` and `version`. Vendor-specific
extensions SHOULD use the `x_` prefix.

---

## Appendix A: Minimal example

A 1-chunk container holding a 1-frame 8×8×1 dataset:

```
mini.gndc:
├── meta.json
│   {
│     "format": "gndc",
│     "version": "0.9",
│     "shape": [1, 8, 8, 1],
│     "axes":  ["time", "y", "x", "band"],
│     "dtype": "uint8",
│     "bands": [
│       {"name": "B1", "scale_factor": 1.0, "add_offset": 0.0,
│        "valid_range": [0, 255]}
│     ],
│     "crs": null,
│     "transform": null,
│     "time": {"values": ["2024-01-01T00:00:00Z"], "calendar": "gregorian"}
│   }
├── manifest.json
│   {
│     "format": "gndc", "version": "0.9",
│     "chunk_grid": {
│       "time": {"n": 1, "regular": true, "size": 1},
│       "y":    {"n": 1, "regular": true, "size": 8},
│       "x":    {"n": 1, "regular": true, "size": 8}
│     },
│     "chunks": [{
│       "id": [0, 0, 0],
│       "bounds": {"time": [0, 1], "y": [0, 8], "x": [0, 8]},
│       "file": "chunks/t0000_y0000_x0000.gndcc",
│       "size_bytes": 4096
│     }]
│   }
└── chunks/
    └── t0000_y0000_x0000.gndcc       (binary, structure per §6)
```

The container opens with `pygndc.open("mini.gndc")` and exposes a
4-D dataset; `ds.read(time=0)` returns an `(8, 8, 1)` float32 array.
