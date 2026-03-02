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

## Usage

Because this plugin follows LeRobot's standard naming conventions, it is automatically discovered by the CLI. You can reference it using `--robot.type=so101_mujoco`.

**Test the simulation with keyboard teleoperation:**

```bash
lerobot-teleoperate --robot.type=so101_mujoco --teleop.type=keyboard

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
