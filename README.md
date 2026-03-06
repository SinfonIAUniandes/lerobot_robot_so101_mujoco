# SO101 MuJoCo Simulation for LeRobot

A custom hardware plugin for the [LeRobot](https://github.com/huggingface/lerobot) framework that wraps a MuJoCo simulation of the SO101 robotic arm. 

This package allows you to use a simulated environment exactly as if it were physical hardware, making it fully compatible with LeRobot's native data collection, teleoperation, and training pipelines. It includes a native MuJoCo viewer that displays the ground-truth reality of the simulated world, alongside virtual RGB and Depth cameras.

## Installation

**Important:** To prevent dependency conflicts with your Python environment (especially concerning OpenCV and specific LeRobot extras), dependencies are managed via `requirements.txt` rather than `pyproject.toml`.

**1. Clone the repository**
```bash
git clone <your-repo-url>
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