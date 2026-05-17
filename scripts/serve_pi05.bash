#!/usr/bin/env bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openpi

cd openpi

# rand_obj
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8000 policy:checkpoint --policy.config=bimanual_microwave_cook --policy.dir=../checkpoints/pi05_dexjoco_ckpt/bimanual_microwave_cook/bimanual_microwave_cook20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8001 policy:checkpoint --policy.config=bimanual_assembly --policy.dir=../checkpoints/pi05_dexjoco_ckpt/bimanual_assembly/bimanual_assembly20260409/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8002 policy:checkpoint --policy.config=bimanual_hanoi --policy.dir=../checkpoints/pi05_dexjoco_ckpt/bimanual_hanoi/bimanual_hanoi20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8003 policy:checkpoint --policy.config=bimanual_photograph --policy.dir=../checkpoints/pi05_dexjoco_ckpt/bimanual_photograph/bimanual_photograph20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8004 policy:checkpoint --policy.config=bimanual_unlock_ipad --policy.dir=../checkpoints/pi05_dexjoco_ckpt/bimanual_unlock_ipad/bimanual_unlock_ipad20260409/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8005 policy:checkpoint --policy.config=click_mouse --policy.dir=../checkpoints/pi05_dexjoco_ckpt/click_mouse/click_mouse20260401/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8006 policy:checkpoint --policy.config=fold_glasses --policy.dir=../checkpoints/pi05_dexjoco_ckpt/fold_glasses/fold_glasses20260401/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8007 policy:checkpoint --policy.config=hammer_nail --policy.dir=../checkpoints/pi05_dexjoco_ckpt/hammer_nail/hammer_nail20260401/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8008 policy:checkpoint --policy.config=pick_bucket --policy.dir=../checkpoints/pi05_dexjoco_ckpt/pick_bucket/pick_bucket20260401/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8009 policy:checkpoint --policy.config=pinch_tongs --policy.dir=../checkpoints/pi05_dexjoco_ckpt/pinch_tongs/pinch_tongs20260330/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8010 policy:checkpoint --policy.config=water_plant --policy.dir=../checkpoints/pi05_dexjoco_ckpt/water_plant/water_plant20260405/29999

# rand_full
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8000 policy:checkpoint --policy.config=bimanual_microwave_cook_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/bimanual_microwave_cook_rand_full/bimanual_microwave_cook_randomize20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8001 policy:checkpoint --policy.config=bimanual_assembly_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/bimanual_assembly_rand_full/bimanual_assembly_randomize20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8002 policy:checkpoint --policy.config=bimanual_hanoi_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/bimanual_hanoi_rand_full/bimanual_hanoi_randomize20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8003 policy:checkpoint --policy.config=bimanual_photograph_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/bimanual_photograph_rand_full/bimanual_photograph_randomize20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8004 policy:checkpoint --policy.config=bimanual_unlock_ipad_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/bimanual_unlock_ipad_rand_full/bimanual_unlock_ipad_randomize20260413/59999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8005 policy:checkpoint --policy.config=click_mouse_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/click_mouse_rand_full/click_mouse_randomize20260405/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8006 policy:checkpoint --policy.config=fold_glasses_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/fold_glasses_rand_full/fold_glasses_randomize20260405/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8007 policy:checkpoint --policy.config=hammer_nail_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/hammer_nail_rand_full/hammer_nail_randomize20260405/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8008 policy:checkpoint --policy.config=pick_bucket_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/pick_bucket_rand_full/pick_bucket_randomize20260405/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8009 policy:checkpoint --policy.config=pinch_tongs_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/pinch_tongs_rand_full/pinch_tongs_randomize20260405/29999
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 python ./scripts/serve_policy.py --port=8010 policy:checkpoint --policy.config=water_plant_rand_full --policy.dir=../checkpoints/pi05_dexjoco_rand_full_ckpt/water_plant_rand_full/water_plant_randomize20260405/29999
