# Reproducibility

[Back to the README](../README.md#replay-success-rate)

## Why replay can differ between machines

Identical code, data and Python packages do not necessarily produce identical
replay. NumPy/SciPy use OpenBLAS, while functions such as `asin` also depend on the
system's **glibc/libm**. Small floating-point differences can accumulate through
the controller and contact simulation, eventually changing task success. A Python
virtual environment does not isolate these system libraries.

In our validation, aligning glibc/libm to **2.39** while retaining **Haswell and
single-thread BLAS** restored **2,200/2,200 successful replays**, with all recorded
observation states matching exactly. Switching to another BLAS kernel still caused
failures. Both the system math runtime and the actual BLAS kernel therefore matter.

## Reference configuration

Use these values when comparing environments. Matching CPU models is not required;
matching the numerical runtime, code, models and data is the priority.

| Component | Validated configuration |
| --- | --- |
| Data | official `DexJoCo-Datasets-v1.1`, both groups, 2,200 episodes |
| Platform | Linux x86_64; Ryzen 9 9950X and EPYC 7763; Haswell-compatible AVX2/FMA support |
| Python | 3.12.3 on the source host; 3.12.16 on the reproduction host |
| glibc/libm | 2.39; locally tested package `libc6_2.39-0ubuntu8.9_amd64.deb` |
| NumPy / SciPy | 2.2.6 / 1.18.1 |
| Bundled OpenBLAS | NumPy: 0.3.29, ILP64; SciPy: 0.3.31.dev, LP64; both **Haswell, 1 thread** |
| MuJoCo | Package and runtime 3.4.0 |
| Transformations | dm-robotics-transformations 0.10.0, `_transformations` backend; optional `numpy-quaternion` absent on reproduction host |
| LeRobot / Gymnasium | 0.7.0 / 1.3.0 |
| Data loading | datasets 4.8.5 / PyArrow 26.0.0 |
| Actions / initialization | Quaternion actions preserved as float64; restore saved initial states |
| Replay | Seed 0, extra steps 60, 10 async environments per task; rendering and both visual/dynamics randomization disabled |
| Model settings | Control/physics timestep 0.02/0.002 s; Euler, Newton, pyramidal cone; iterations/tolerance 100/1e-8; remaining options unchanged from the code revision |
| Floating-point mode | Round-to-nearest; FTZ and DAZ off; no custom `GLIBC_TUNABLES` |

Set the kernel and thread variables **before importing NumPy/SciPy**. DexJoCo sets
Haswell and the NumPy SIMD exclusions when imported, but that can be too late if
another library imports NumPy first. Check actual loaded libraries and thread
counts, rather than relying only on environment-variable values.

## Use a matching runtime without changing project code

Keep the Python dependencies and model/data versions aligned with the reference.
If the host glibc differs, the following optional setup loads a private copy for
replay. It requires **Linux x86_64, Bash, curl, dpkg-deb and an existing `.venv`**.
All generated files stay in the ignored `.runtime/` directory; no system library,
application source, dependency lockfile or shell profile is changed.

**1. From the repository root, download, verify and extract the tested package.**
For offline use, place the same `.deb` at the archive path before running this block.

```bash
(
  set -euo pipefail
  mkdir -p .runtime/replay .runtime/glibc-2.39-0ubuntu8.9
  archive=.runtime/glibc-2.39-0ubuntu8.9/libc6.deb
  if [ ! -f "$archive" ]; then
    curl -fL --retry 3 \
      https://archive.ubuntu.com/ubuntu/pool/main/g/glibc/libc6_2.39-0ubuntu8.9_amd64.deb \
      -o "$archive"
  fi
  echo "ff5557d99b51f761c4b7c92368b9cc45565eda17df9bf9eb4b134d09825008be  $archive" | sha256sum -c -
  dpkg-deb -x "$archive" .runtime/glibc-2.39-0ubuntu8.9/root
)
```

**2. Create a local entry point.** The startup hook also sends Python workers
through this entry point, so async replay uses the same runtime as its parent.

```bash
cat > .runtime/replay/python <<'SH'
#!/usr/bin/env bash
set -euo pipefail
local_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$local_dir/../.." && pwd)
libs="$repo/.runtime/glibc-2.39-0ubuntu8.9/root/usr/lib/x86_64-linux-gnu"
python="$repo/.venv/bin/python"
python_lib=$(dirname -- "$(readlink -f -- "$python")")/../lib
export OPENBLAS_CORETYPE=HASWELL
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
export DEXJOCO_REPRO_EXECUTABLE="$local_dir/python"
export PYTHONPATH="$local_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$libs/ld-linux-x86-64.so.2" \
  --library-path "$libs:$python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$python" "$@"
SH
chmod +x .runtime/replay/python
cat > .runtime/replay/sitecustomize.py <<'PY'
import ctypes
import multiprocessing
import os
import sys

libc = ctypes.CDLL(None)
libc.gnu_get_libc_version.restype = ctypes.c_char_p
if libc.gnu_get_libc_version() != b"2.39":
    print("Expected glibc 2.39; use .runtime/replay/python", file=sys.stderr, flush=True)
    os._exit(1)
sys.executable = os.environ["DEXJOCO_REPRO_EXECUTABLE"]
multiprocessing.set_executable(sys.executable)
PY
```

This hook applies only to processes started through the local entry point and
occupies `sitecustomize` for them. Use it with scripts, `-m` or `-c`; **do not use
`-I`, `-S` or `-E`**, which bypass the hook. Subprocesses using `sys.executable`
inherit the entry point; hard-coded Python paths do not. Preserve your normal
[FFmpeg setup](install_ffmpeg.md) when decoding videos.

**3. Check the runtime, then replay a dataset.** Replace the dataset path below.

```bash
.runtime/replay/python -c \
  'import ctypes; c=ctypes.CDLL(None); c.gnu_get_libc_version.restype=ctypes.c_char_p; print(c.gnu_get_libc_version().decode())'
# Expected: 2.39
.runtime/replay/python .venv/bin/dexjoco-replay \
  --dataset-dir /path/to/DexJoCo-Datasets-v1.1/rand_obj/click_mouse \
  --task click_mouse --action-format quaternion \
  --n-envs 10 --use-async-envs --extend-steps 60 --seed 0
# Expected: Replay success: 100/100
```

Use the same prefix for `.venv/bin/lerobot-eval` or `.venv/bin/lerobot-train` with their normal arguments. **Activating `.venv` alone still uses the host glibc.**
To stop using the private runtime, run Python normally.

If replay still differs, compare actual glibc/libm and OpenBLAS versions/kernel/thread counts first, then the reference dependencies, transformation backend, actions, initial states and model options.
The replay CLI checks task success; exact trajectory matching additionally requires
comparison against the dataset's stored `observation.state`.

## Anti-aliasing and rendering

All task XMLs default to **`offsamples=0`**, disabling offscreen multisample anti-aliasing (MSAA). Replay and recording accept `--offsamples`; omitting it preserves the XML setting. Add `--offsamples 4` to record with four samples per pixel.

EGL rendering with MSAA enabled can produce small pixel differences between fresh rendering contexts. A visual policy can amplify these differences into different actions and success rates. **Use matching anti-aliasing settings for dataset recording and policy evaluation.** The v1.1 640 × 640 replay datasets were recorded with **MSAA disabled**. See [issue #24](https://github.com/brave-eai/dexjoco/issues/24).

Dataset videos use **lossy compression** to keep storage requirements manageable. Training therefore receives decoded video frames, while online inference receives freshly rendered frames. Even with identical rendering settings, these images can differ at the pixel level. Lossless storage would substantially increase dataset size.
