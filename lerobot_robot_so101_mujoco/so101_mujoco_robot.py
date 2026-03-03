import os
import threading
import numpy as np
from typing import Any
from lerobot.robots.robot import Robot
import time

# Import your existing simulation class
from .so101_sim import SO101Simulation
from .config_so101_mujoco_robot import So101MujocoRobotConfig

class So101MujocoRobot(Robot):
    # The device class name must match the config class name without the Config suffix
    config_class = So101MujocoRobotConfig
    name = "so101_mujoco"

    def __init__(self, config: So101MujocoRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        
        # Thread-safe storage for observations and target actions
        self._latest_obs = {}
        self._target_action = {}
        self._sim_thread = None

        self.cameras = {"camera": None}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_xml_path = os.path.join(current_dir, "robotstudio_so101", "so101_camera_mount.xml")

        # Instantiate the simulation but DO NOT run it yet
        self.sim = SO101Simulation(
            xml_path=resolved_xml_path,
            urdf_name=self.config.urdf_name,
            enable_rgb=True,
            show_cv2=False, # Let LeRobot handle any visualizations
            rgb_callback=self._on_rgb_frame,
            joint_callback=self._on_joint_data,
            control_callback=self._on_control_request
        )

    # --- Callbacks for the background thread ---
    def _on_rgb_frame(self, bgr_image):
        import cv2
        # Convert the BGR image from the sim to RGB for LeRobot
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self._latest_obs["camera"] = rgb_image

    def _on_joint_data(self, joint_data):
        # Flatten joint data into the format LeRobot expects
        for joint_name, pos in joint_data.items():
            self._latest_obs[f"{joint_name}.pos"] = pos

    def _on_control_request(self, sim_time):
        # The simulation thread calls this to get the latest commanded action
        return self._target_action

    # --- LeRobot Interface Implementation ---
    @property
    def observation_features(self) -> dict:
        # Define the structure of sensor outputs
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float, 
            "camera": (480, 640, 3) 
        }

    @property
    def action_features(self) -> dict:
        # Define the commands your robot expects via send_action()
        return {
            "shoulder_pan": float,
            "shoulder_lift": float,
            "elbow_flex": float,
            "wrist_flex": float,
            "wrist_roll": float,
            "gripper": float,
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        # Simulations start perfectly aligned
        return True

    def calibrate(self) -> None:
        # No physical offsets to calculate
        pass

    def configure(self) -> None:
        # No hardware PID or torque settings to initialize
        pass

    def connect(self, calibrate: bool = True) -> None:
        # Pass headless=True using a lambda to avoid the GLFW/Wayland crash
        self._sim_thread = threading.Thread(target=lambda: self.sim.run(headless=True), daemon=True)
        self._sim_thread.start()
        self._is_connected = True

        for key in self.action_features.keys():
            self._target_action[key] = 0.0
            
        print("Waiting for MuJoCo to render the first frame...")
        while "camera" not in self._latest_obs or "gripper.pos" not in self._latest_obs:
            time.sleep(0.05)
        print("MuJoCo is ready!")

        self._is_connected = True

    def disconnect(self) -> None:
        # Gracefully terminate the simulation loop
        self._is_connected = False
        if hasattr(self, 'sim'):
            self.sim.is_running = False

    def get_observation(self) -> dict[str, Any]:
        # Return a dictionary of sensor values from the robot
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        return self._latest_obs.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Update the target action dictionary for the sim thread to pick up
        for key, val in action.items():
            self._target_action[key] = val
        return action