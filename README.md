# SO101 MuJoCo Simulation for LeRobot

A custom hardware plugin for the [LeRobot](https://github.com/huggingface/lerobot) framework that wraps a MuJoCo simulation of the SO101 robotic arm. 

This package allows you to use a simulated environment exactly as if it were physical hardware, making it fully compatible with LeRobot's native data collection, teleoperation, and training pipelines. It includes a native MuJoCo viewer that displays the ground-truth reality of the simulated world, alongside virtual RGB and Depth cameras.

## Installation

**Important:** To prevent dependency conflicts with your Python environment (especially concerning OpenCV and specific LeRobot extras), dependencies are managed via `requirements.txt` rather than `pyproject.toml`.

**1. Clone the repository**
```bash
git clone https://github.com/SinfonIAUniandes/lerobot_robot_so101_mujoco
cd lerobot_robot_so101_mujoco

```

**2. Install the strict dependencies**
Ensure you are in your desired virtual environment, then install the required packages:

```bash
pip install -r requirements.txt

```

**3. Install the plugin**
Install the package in editable mode so it registers with the LeRobot CLI:

```bash
pip install -e .

```

## Configuration Parameters

You can customize the simulation and telemetry output directly from the LeRobot CLI by appending `--robot.<parameter_name>=<value>`.

| Parameter | Default | Description |
| --- | --- | --- |
| `xml_path` | `"./robotstudio_so101/so101_camera_mount.xml"` | Path to your MuJoCo XML scene definition. |
| `camera_name` | `"realsense_d435i"` | The name of the camera defined in the XML to use for RGB/Depth rendering. |
| `render_fps` | `30` | The target frames-per-second for the simulated camera sensors. |
| `enable_rgb` | `True` | Streams the RGB camera array to the LeRobot pipeline. |
| `enable_depth` | `False` | Renders the depth map (useful if sending pointclouds to Rerun). |
| `show_cv2` | `False` | Opens a pop-up OpenCV window showing the live camera feeds. |
| `enable_rerun` | `False` | Enables live telemetry streaming to the Rerun.io visualizer. |
| `rerun_log_meshes` | `True` | Logs the static 3D meshes of the robot to Rerun. |
| `rerun_log_tf` | `True` | Logs dynamic TF trees and joint frames to Rerun. |
| `rerun_depth_mode` | `"none"` | Set to `"pointcloud"` to stream a live 3D pointcloud projection of the depth camera to Rerun. |

## Usage

Because this plugin follows LeRobot's standard naming conventions, it is automatically discovered by the CLI. You can reference it using `--robot.type=so101_mujoco`.

**Test the simulation with keyboard teleoperation and Rerun telemetry:**

```bash
lerobot-teleoperate \
  --robot.type=so101_mujoco \
  --robot.enable_rerun=true \
  --robot.rerun_depth_mode=pointcloud \
  --teleop.type=keyboard

```

**Record a simulated dataset:**

```bash
lerobot-record \
  --robot.type=so101_mujoco \
  --teleop.type=keyboard \
  --dataset.repo_id=local/so101_sim_data \
  --dataset.single_task="reach the target" \
  --dataset.fps=30

```


**Example: override all configuration parameters from the CLI**

You can override any configuration parameter by passing `--robot.<parameter_name>=<value>` to the LeRobot command. Below is an example that shows how to modify all parameters defined in the `So101MujocoRobotConfig` in a single command (use `\` to split the line for readability in bash):

```bash
lerobot-teleoperate \
  --robot.type=so101_mujoco \
  --robot.xml_path="./custom_scene/so101_custom.xml" \
  --robot.camera_name="my_camera" \
  --robot.render_fps=15 \
  --robot.enable_rgb=false \
  --robot.enable_depth=true \
  --robot.show_cv2=true \
  --robot.enable_rerun=true \
  --robot.rerun_log_meshes=false \
  --robot.rerun_log_tf=false \
  --robot.rerun_depth_mode=pointcloud \
  --robot.rerun_log_rgb=false \
  --robot.randomize_scene=false \
  --robot.box_pos_base="(0.40, 0.05, 0.02)" \
  --robot.box_pos_delta="(0.02, 0.02, 0.00)" \
  --robot.box_size_base="(0.03, 0.03, 0.04)" \
  --robot.box_size_delta="(0.005, 0.005, 0.005)" \
  --robot.box_color_base="(0.2, 0.2, 0.2, 1.0)" \
  --robot.box_color_delta="(0.05, 0.05, 0.05, 0.0)" \
  --robot.tray_pos_base="(0.36, 0.22, 0.01)" \
  --robot.tray_pos_delta="(0.05, 0.03, 0.00)" \
  --robot.tray_size_base="(0.09, 0.07, 0.01)" \
  --robot.tray_size_delta="(0.01, 0.01, 0.00)" \
  --robot.tray_color_base="(0.9, 0.2, 0.2, 1.0)" \
  --robot.tray_color_delta="(0.1, 0.05, 0.05, 0.0)" \
  --robot.camera_pos_base="(0.25, -0.35, 0.55)" \
  --robot.camera_pos_delta="(0.00, 0.00, 0.00)" \
  --robot.camera_euler_base="(0.70, 3.14159, 0.0)" \
  --robot.camera_euler_delta="(0.00, 0.00, 0.00)" \
  --teleop.type=keyboard
```

