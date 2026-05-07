# Geometric Retargeting

[![CC BY-NC 4.0 License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

Welcome! This repository contains the code for the paper "Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm".

![Demo GIF](./images/demo.gif)
## Installation
If you have already got a conda environment with torch, you just need these packages
```
pip install trimesh open3d sapien zmq
pip install -e .
```

Large checkpoints and the MediaPipe hand landmark model are distributed separately from Git. From the Dexjoco repository root, restore the pieces you need with:

```
python scripts/download_assets.py --bundle geort-runtime-assets
```

Otherwise, we recommend using a virtual environment to install the required packages. To install the required packages, run the following command:
```
conda create --name geort python=3.8
pip install -r requirements.txt
pip install -e .
```
## Quick Overview
Upon completion, you will be able to train GeoRT and deploy the checkpoint in a clean and straightforward way. 
### Training (1-2min):
```
python ./geort/trainer.py -hand allegro_right -human_data YOUR_DATASET_NAME -ckpt_tag geort_1
```
### Deploy in code
```
import geort
model = geort.load_model('geort_1')
mocap = ...
qpos = model.forward(mocap.get())
```
But before this, we need to complete some one-time system setup steps outlined below.

**Useful Links**: [Notes and Troubleshooting](#notes-and-troubleshooting)
## Getting Started
We use the native Allegro Hand as an example. 

### Step 1: Import your robot hand (one-time setup).
Note: For the Allegro Hand, you can actually skip this step. However, please follow it if you want to import a customized robot hand.

We just need to complete a quick setup process outlined below:

1. Place your robot hand URDF file in the ``assets`` folder. (We have included the Allegro example there.)
2. Create a config file named ``your_robot_name.json`` in the ``geort/config`` directory. Below is an example for the Allegro hand. For brevity, the details are omitted here, but you can refer to the [this](./geort/config/allegro_right.json) for full information. For setup instructions, please read [this](./geort/config/template.py).

```
{
    "name": "allegro_right",  
    "urdf_path": "./assets/allegro_right/allegro_hand_description_right.urdf",
    "base_link": "base_link",
    "joint_order": [
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"
    ],
    "fingertip_link": [
        {
            "name": "index",
            "link": "link_4.0_tip",
            "joint": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            "center_offset": [0.0, 0.0, 0.0],
            "human_hand_id": 8,
        },
        ...
    ]
}

```
Now, you can run this command to visualize your hand.
```
python geort/env/hand.py --hand [YOUR_HAND_CONFIG_NAME]
```
such as 
```
python geort/env/hand.py --hand allegro_right
```
<span style="color:red"> If there is any segmentation error, please simplify the collision meshes or just remove all the `<collision>` fields in your URDF. </span> See the [Notes and Troubleshooting](#notes-and-troubleshooting) section.

### Step 2: Collect human hand mocap data.
Now we need to collect some human hand data for training the retargeting model. We put an example human recording dataset in data folder. You can add your own data to that folder and here is a template python script to do this.

```
import geort
import time

# Dataset Name
data_output_name = "human" # TODO(): Specify a name for this (e.g. your name)

# Your data collection loop.
mocap = YourAwesomeMocap() # TODO(): your mocap system.
                           # Define a mocap.get() method.
                           # Apologies, you still have to do this...
 
data = []

for step in range(5000):       # collect 5000 data points.
    hand_keypoint = mocap.get() # mocap.get() return [N, 3] numpy array.
    data.append(hand_keypoint)
    
    time.sleep(0.01)            # take a short break.

# finish data collection.
geort.save_human_data(data, data_output_name)
```
Use ``geort.save_human_data`` API -- this can simplify your effort in specifying the path. This dataset can be reloaded later using **data_output_name**. 

During the data collection process, try to 1. fully stretch each finger and explore its fingertip moving range and 2. perform pinch grasps. Ensure that your fingers feel natural and comfortable—since during teleoperation deployment, you will use these recorded gestures to control the robot! Please avoid any unnatural or strained movements.

We understand that most users likely have their own mocap systems. However, for demonstration purposes, we provide a simple mocap solution based on MediaPipe. Please note, this is intended only for demo use and not for deployment; we will explain this in more detail later.

```
python ./geort/mocap/mediapipe_mocap.py --name human
```
to generate a dataset named ``human``. Refered to the file for instructions. When you see the pop-up window, press ``s`` to start recording and ``q`` to finish. 

**Note:** Please ensure that the hand frame orientation is consistent between your motion capture system and the hand URDF (but fortunately the origin does not require any alignment and you can just set it to palm center). In our provided mocap example, we support the **right** hand using the following convention:+Y axis: from the palm center to the thumb. +Z axis: from the palm center to the middle fingertip. +X axis: palm normal (pointing out of the palm). 

### Step 3: Train the Model
Assuming you have placed ``your_robot_name.json`` in the ``geort/config`` folder as described in Step 1, and set ``data_output_name`` to ``human`` in Step 2, run the following command. TAG is the checkpoint id to use in later deployment.

```
python ./geort/trainer.py -hand your_robot_name -human_data human -ckpt_tag TAG
```

Let it train for about 30–50 epochs (approximately 1–2 minutes). You can press Ctrl+C to stop early if you wish. 

If this is the first time you’re training for a new hand, an additional 5 minutes will be needed to train the neural FK model — this only happens once.
In the command above, 

In this Dexjoco release, the ``data`` folder is intentionally kept empty. Place your own recorded hand data there and reference it by name when training.

### Step 4: Deploy!
Ok, now we are all set. Use the following code to import and deploy the trained model. 

```
import geort

checkpoint_tag = 'geort_1'          # TODO: your checkpoint name, assume it is 'TAG'
model = geort.load_model(checkpoint_tag, epoch=50)  # set epoch=-1 to use the last model.

mocap = YourAwesomeMocap()      # TODO: your mocap.
robot = YourRobustRobotHand()   # TODO: your robot.

while True:
    qpos = model.forward(mocap.get()) # This is the retargeted qpos. 
                                      # (Note: unnormalized joint angle)
    robot.command(qpos)               # execute!

```
We provide some examples in ``geort/mocap/mediapipe_evaluation.py`` and ``geort/mocap/replay_evaluation.py``. We recommend (insist) you use a glove-based or tracker-based mocap system instead of MediaPipe, as for vision-based mocap there is significant input distribution shift during deployment.

The simplest way for testing is to use the replay evaluation as below. This will show the retargeted trajectory in the viewer. 
```
python ./geort/mocap/replay_evaluation.py -hand allegro_right -ckpt_tag YOUR_CKPT -data YOUR_TRAINING_DATA
```
For instance, if you have ``human.npy`` in the ``data`` folder
```
python ./geort/mocap/replay_evaluation.py -hand allegro_right -ckpt_tag YOUR_CKPT -data human
```
## Contributing
Feel free to contribute your robot model and mocap system to the GeoRT repository!

## [Notes and Troubleshooting](#notes-and-troubleshooting)
1. **Note:Joint Range Clipping.** One core assumption of GeoRT is that the motion range of robot fingertips resembles that of human hands. To maintain realistic fingertip poses, please clip your robot's joint movement ranges appropriately and avoid unnatural configurations.

2. **Simulation Errors with New Hands?** Simulation errors (segmentation fault) may occur when importing new robotic hands (e.g. [this issue](https://github.com/facebookresearch/GeoRT/issues/7)), and this is usually caused by collision meshes. To avoid this, ensure that the collision meshes defined in your URDF are simple—such as boxes or basic convex shapes. Alternatively, you can remove all <collision> elements from the URDF to eliminate these issues entirely. 

3. **Hand Coordinate System (Frame) Convention** Please ensure that the hand frame orientation is consistent between your motion capture system and the hand URDF (but fortunately the origin does not require any alignment and you can just set it to palm center). In our provided mocap example, we support the **right** hand using the following convention:+Y axis: from the palm center to the thumb. +Z axis: from the palm center to the middle fingertip. +X axis: palm normal (pointing out of the palm). 


## Contact Us
For any inquiries, please open an issue or contact the authors via email at ``zhaohengyin@cs.berkeley.edu``
<!-- ## Bibliography -->

## License
CC-by-NC license
