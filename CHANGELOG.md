# Release notes

## 1.0.14 — 2026-10-05

- Faster encoder startup.
- More efficient GPU training.
- Lower memory use when reading residual corrections for requested dates from
  large archives. Older residual formats remain supported.
- Faster GPU loading of supported quantized models and repeated queries to the
  same date.
- Applications processing chunks in sequence can configure read-ahead and
  preparation budgets through `GNDCDataset`.
- Unreadable residual corrections raise a clear error instead of returning an
  incomplete reconstruction.

Existing `.gndc` files remain supported; re-encoding is not required.

Prebuilt packages cover Python 3.10–3.13 on Windows x86-64 and Linux x86-64
(glibc 2.28 or newer). Reading and decoding do not require an encoder license.
Creating `.gndc` files requires a valid encoder license.
