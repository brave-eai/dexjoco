"""DexJoCo simulation environments."""

import os


def configure_replay_runtime() -> None:
    """Select reproducible CPU kernels before importing NumPy or BLAS."""
    os.environ["OPENBLAS_CORETYPE"] = "HASWELL"
    os.environ["NPY_DISABLE_CPU_FEATURES"] = (
        "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
    )


configure_replay_runtime()
