#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dexjoco

dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_microwave_cook.yaml --seed=0 --port=8000
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_microwave_cook.yaml --seed=1 --port=8000
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_microwave_cook.yaml --seed=2 --port=8000
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_microwave_cook.yaml --seed=0 --port=8000 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_microwave_cook.yaml --seed=1 --port=8000 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_microwave_cook.yaml --seed=2 --port=8000 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_assembly.yaml --seed=0 --port=8001
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_assembly.yaml --seed=1 --port=8001
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_assembly.yaml --seed=2 --port=8001
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_assembly.yaml --seed=0 --port=8001 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_assembly.yaml --seed=1 --port=8001 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_assembly.yaml --seed=2 --port=8001 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_hanoi.yaml --seed=0 --port=8002
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_hanoi.yaml --seed=1 --port=8002
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_hanoi.yaml --seed=2 --port=8002
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_hanoi.yaml --seed=0 --port=8002 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_hanoi.yaml --seed=1 --port=8002 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_hanoi.yaml --seed=2 --port=8002 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_photograph.yaml --seed=0 --port=8003
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_photograph.yaml --seed=1 --port=8003
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_photograph.yaml --seed=2 --port=8003
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_photograph.yaml --seed=0 --port=8003 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_photograph.yaml --seed=1 --port=8003 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_photograph.yaml --seed=2 --port=8003 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_unlock_ipad.yaml --seed=0 --port=8004
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_unlock_ipad.yaml --seed=1 --port=8004
# dexjoco-openpi-eval --config=./configs/rand_obj/bimanual_unlock_ipad.yaml --seed=2 --port=8004
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_unlock_ipad.yaml --seed=0 --port=8004 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_unlock_ipad.yaml --seed=1 --port=8004 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/bimanual_unlock_ipad.yaml --seed=2 --port=8004 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/click_mouse.yaml --seed=0 --port=8005
# dexjoco-openpi-eval --config=./configs/rand_obj/click_mouse.yaml --seed=1 --port=8005
# dexjoco-openpi-eval --config=./configs/rand_obj/click_mouse.yaml --seed=2 --port=8005
# dexjoco-openpi-eval --config=./configs/rand_full/click_mouse.yaml --seed=0 --port=8005 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/click_mouse.yaml --seed=1 --port=8005 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/click_mouse.yaml --seed=2 --port=8005 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/fold_glasses.yaml --seed=0 --port=8006
# dexjoco-openpi-eval --config=./configs/rand_obj/fold_glasses.yaml --seed=1 --port=8006
# dexjoco-openpi-eval --config=./configs/rand_obj/fold_glasses.yaml --seed=2 --port=8006
# dexjoco-openpi-eval --config=./configs/rand_full/fold_glasses.yaml --seed=0 --port=8006 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/fold_glasses.yaml --seed=1 --port=8006 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/fold_glasses.yaml --seed=2 --port=8006 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/hammer_nail.yaml --seed=0 --port=8007
# dexjoco-openpi-eval --config=./configs/rand_obj/hammer_nail.yaml --seed=1 --port=8007
# dexjoco-openpi-eval --config=./configs/rand_obj/hammer_nail.yaml --seed=2 --port=8007
# dexjoco-openpi-eval --config=./configs/rand_full/hammer_nail.yaml --seed=0 --port=8007 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/hammer_nail.yaml --seed=1 --port=8007 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/hammer_nail.yaml --seed=2 --port=8007 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/pick_bucket.yaml --seed=0 --port=8008
# dexjoco-openpi-eval --config=./configs/rand_obj/pick_bucket.yaml --seed=1 --port=8008
# dexjoco-openpi-eval --config=./configs/rand_obj/pick_bucket.yaml --seed=2 --port=8008
# dexjoco-openpi-eval --config=./configs/rand_full/pick_bucket.yaml --seed=0 --port=8008 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/pick_bucket.yaml --seed=1 --port=8008 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/pick_bucket.yaml --seed=2 --port=8008 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/pinch_tongs.yaml --seed=0 --port=8009
# dexjoco-openpi-eval --config=./configs/rand_obj/pinch_tongs.yaml --seed=1 --port=8009
# dexjoco-openpi-eval --config=./configs/rand_obj/pinch_tongs.yaml --seed=2 --port=8009
# dexjoco-openpi-eval --config=./configs/rand_full/pinch_tongs.yaml --seed=0 --port=8009 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/pinch_tongs.yaml --seed=1 --port=8009 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/pinch_tongs.yaml --seed=2 --port=8009 --rand-full

# dexjoco-openpi-eval --config=./configs/rand_obj/water_plant.yaml --seed=0 --port=8010
# dexjoco-openpi-eval --config=./configs/rand_obj/water_plant.yaml --seed=1 --port=8010
# dexjoco-openpi-eval --config=./configs/rand_obj/water_plant.yaml --seed=2 --port=8010
# dexjoco-openpi-eval --config=./configs/rand_full/water_plant.yaml --seed=0 --port=8010 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/water_plant.yaml --seed=1 --port=8010 --rand-full
# dexjoco-openpi-eval --config=./configs/rand_full/water_plant.yaml --seed=2 --port=8010 --rand-full
