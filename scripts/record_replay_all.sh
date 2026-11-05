#!/usr/bin/env bash
# Record every task from scratch; stop immediately if a task fails.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_dir"

if [[ "${1:-}" == "--help" || $# -gt 2 ]]; then
    echo "Usage: bash scripts/record_replay_all.sh [source_root] [output_root]"
    echo "Defaults: datasets/lerobot_v1.1 -> datasets/lerobot_v1.1_replay_640"
    echo "Environment: N_ENVS=2, SEED=0 (visual randomization base seed)."
    exit 0
fi

source_root="$(realpath -m -- "${1:-datasets/lerobot_v1.1}")"
output_root="$(realpath -m -- "${2:-datasets/lerobot_v1.1_replay_640}")"
n_envs="${N_ENVS:-2}"
seed="${SEED:-0}"

export OPENBLAS_CORETYPE=HASWELL
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export PYTHONUNBUFFERED=1

tasks=(
    water_plant fold_glasses click_mouse pinch_tongs pick_bucket hammer_nail
    bimanual_microwave_cook bimanual_unlock_ipad bimanual_hanoi
    bimanual_assembly bimanual_photograph
)
groups=(rand_obj rand_full)

if [[ -e "$output_root" ]]; then
    echo "Output already exists: $output_root" >&2
    echo "Choose a new output directory for a complete run." >&2
    exit 1
fi
for group in "${groups[@]}"; do
    for task in "${tasks[@]}"; do
        if [[ ! -f "$source_root/$group/$task/meta/info.json" ]]; then
            echo "Missing source dataset: $source_root/$group/$task" >&2
            exit 1
        fi
    done
done

mkdir -p -- "$output_root/logs"
completed=0
for group in "${groups[@]}"; do
    randomize=--no-randomize
    if [[ "$group" == rand_full ]]; then
        randomize=--randomize
    fi
    for task in "${tasks[@]}"; do
        echo "[$((completed + 1))/22] Recording $group/$task"
        uv run --no-sync dexjoco-record-replay \
            --dataset-dir "$source_root/$group/$task" \
            --output-dir "$output_root/$group/$task" \
            --task "$task" \
            --render-hw 640 640 \
            --n-envs "$n_envs" \
            --use-async-envs \
            --action-format quaternion \
            --seed "$seed" \
            "$randomize" \
            2>&1 | tee "$output_root/logs/${group}_${task}.log"
        completed=$((completed + 1))
    done
done
echo "Finished all $completed datasets: $output_root"
