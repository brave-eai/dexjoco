# Geometric Retargeting for DexJoCo

DexJoCo's default GeoRT retarget checkpoints are tracked directly in this repository under `teleoperation/GeoRT/checkpoint/`.

## Installation

We recommend using a virtual environment to install the required packages. To install the required packages, run the following command:

```bash
conda create --name geort python=3.8
pip install -r requirements.txt
pip install -e .
```

## Getting Started

### Step 1: Collect Human Hand Mocap Data

You need to collect human hand data to train the retargeting model. Follow the tutorial to configure Rokoko so that it can stream pose data to the PC.

```bash
python dexjoco/teleoperation/rokoko/collect_mocap_data.py \
  --listen-ip <LISTEN_IP> \
  --listen-port <ROKOKO_STREAMING_PORT> \
  --hand <HAND_TYPE> \
  --output-name <OUTPUT_NAME>
```

The collected data should be placed under:

```text
./GeoRT/data/
```

During data collection, try to:

1. Fully stretch each finger and explore its fingertip range of motion.
2. Perform pinch grasps.

Ensure that your fingers feel natural and comfortable, because during teleoperation deployment, you will use these recorded gestures to control the robot. Avoid any unnatural or strained movements.

### Step 2: Train the Model

```bash
python ./geort/trainer.py -hand allegro_right -human_data YOUR_DATASET_NAME -ckpt_tag TAG
```

Let it train for about 30-50 epochs (approximately 1-2 minutes). You can press `Ctrl+C` to stop early if you wish.

### Step 3: Deploy

We provide deployment examples in `geort/mocap/rokoko_evaluation.py`.

The simplest way to test the trained retargeting network is to run replay evaluation. This will visualize the retargeted hand trajectory in the viewer:

```bash
python ./geort/mocap/rokoko_evaluation.py \
  -hand allegro_right \
  -ckpt_tag <YOUR_CKPT> \
  -data <YOUR_TRAINING_DATA>
```

Once the test runs successfully, you can use the trained retargeting network for teleoperation and collect robot hand trajectories in the simulator.

Before running the retargeting scripts, you need to run the Rokoko mocap sender on the PC where Rokoko Studio is installed. This script receives the hand pose data streamed from Rokoko Studio and forwards the left- and right-hand data to the corresponding retargeting ports.

```bash
python ../rokoko/rokoko_mocap_bimanual.py \
  --listen-ip <LISTEN_IP> \
  --listen-port <ROKOKO_STREAMING_PORT> \
  --target-ip <TARGET_IP> \
  --left-port <LEFT_HAND_PORT> \
  --right-port <RIGHT_HAND_PORT>
```

Make sure that the listening port and the target ports are correctly configured and consistent with your simulator setup.

To retarget left-hand motion from Rokoko to the left Allegro Hand, run:

```bash
python ./geort/mocap/rokoko_retarget_send_left.py \
  --bind_ip <BIND_IP> \
  --bind_port <BIND_PORT> \
  --target_ip <TARGET_IP> \
  --target_port <TARGET_PORT>
```

Default values:

```text
--bind_port 5015
--target_port 5016
```

To retarget right-hand motion from Rokoko to the right Allegro Hand, run:

```bash
python ./geort/mocap/rokoko_retarget_send_right.py \
  --bind_ip <BIND_IP> \
  --bind_port <BIND_PORT> \
  --target_ip <TARGET_IP> \
  --target_port <TARGET_PORT>
```

Default values:

```text
--bind_port 5013
--target_port 5014
```

DexJoCo's simulator receives these packets in [`../../dexjoco/dexjoco/tasks/sim_teleop.py`](../../dexjoco/dexjoco/tasks/sim_teleop.py).
