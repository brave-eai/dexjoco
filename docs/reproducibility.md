# Reproducibility

[Back to the README](../README.md#replay-success-rate)

## CPU configuration

**`dexjoco-replay` and `dexjoco-record-replay` automatically set the following environment variables before importing NumPy**, selecting the Haswell OpenBLAS kernel and disabling selected AVX-512 features for reproducibility. No manual exports are needed for these commands. DexJoCo replaces any existing values for these variables when its package is imported.

When integrating replay into a Python process that imports NumPy first, **set these variables before starting Python**:

```bash
export OPENBLAS_CORETYPE=HASWELL
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
```

## Anti-aliasing and rendering

All task XMLs default to **`offsamples=0`**, disabling offscreen multisample anti-aliasing (MSAA). Replay and recording accept `--offsamples`; omitting it preserves the XML setting. Add `--offsamples 4` to record with four samples per pixel.

EGL rendering with MSAA enabled can produce small pixel differences between fresh rendering contexts. A visual policy can amplify these differences into different actions and success rates. **Use matching anti-aliasing settings for dataset recording and policy evaluation.** The v1.1 640 × 640 replay datasets were recorded with **MSAA disabled**. See [issue #24](https://github.com/brave-eai/dexjoco/issues/24).

Dataset videos use **lossy compression** to keep storage requirements manageable. Training therefore receives decoded video frames, while online inference receives freshly rendered frames. Even with identical rendering settings, these images can differ at the pixel level. Lossless storage would substantially increase dataset size.
