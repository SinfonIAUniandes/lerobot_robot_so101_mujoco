import os
import threading
from typing import Any
from lerobot.robots.robot import Robot
import time
import numpy as np
import cv2

from .so101_sim import SO101Simulation
from .config_so101_mujoco_robot import So101MujocoRobotConfig


class So101MujocoRobot(Robot):
    config_class = So101MujocoRobotConfig
    name = "so101_mujoco"

    def __init__(self, config: So101MujocoRobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        self._latest_obs = {}
        self._target_action = {}
        self._sim_thread = None
        self.scale = 50.0

        self.cameras = {"realsense": None}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_xml_path = os.path.join(current_dir, self.config.xml_path)

        # Instantiate the simulation with the new config parameters
        self.sim = SO101Simulation(
            xml_path=resolved_xml_path,
            camera_name=self.config.camera_name,
            render_fps=self.config.render_fps,
            enable_rgb=self.config.enable_rgb,
            enable_depth=self.config.enable_depth,
            show_cv2=self.config.show_cv2,
            enable_rerun=self.config.enable_rerun,
            rerun_log_meshes=self.config.rerun_log_meshes,
            rerun_log_tf=self.config.rerun_log_tf,
            rerun_depth_mode=self.config.rerun_depth_mode,
            rerun_log_rgb=self.config.rerun_log_rgb,
            wrist_callback=self._on_wrist_frame if self.config.enable_wrist_cam else None,
            depth_callback=self._on_depth_frame if self.config.enable_depth else None,
            rgb_callback=self._on_rgb_frame,
            joint_callback=self._on_joint_data,
            control_callback=self._on_control_request,
            scene_config=self.config
        )

    def _on_wrist_frame(self, rgb_image):
        self._latest_obs["wrist_cam"] = rgb_image

    def _on_depth_frame(self, raw_depth, depth_colormap):
        # 1. The Pretty Image (for human visualization)
        depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
        self._latest_obs["realsense_depth_vis"] = depth_rgb

        # 2. The Math Data: Encode float32 meters into a 3-channel uint8 image
        # Convert physical meters to millimeters (max range ~65 meters)
        depth_mm = np.clip(raw_depth * 1000, 0, 65535).astype(np.uint16)
        
        # Create a blank 3-channel 8-bit image to trick LeRobot
        encoded_depth = np.zeros((raw_depth.shape[0], raw_depth.shape[1], 3), dtype=np.uint8)
        
        # Pack the 16-bit integer into the Red and Green channels
        encoded_depth[..., 0] = (depth_mm >> 8) & 0xFF  # High byte in Red
        encoded_depth[..., 1] = depth_mm & 0xFF         # Low byte in Green
        # Blue channel stays 0
        
        self._latest_obs["realsense_depth"] = encoded_depth

    def _on_rgb_frame(self, bgr_image):
        import cv2
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self._latest_obs["realsense"] = rgb_image

    def _on_joint_data(self, joint_data):
        for key, value in joint_data.items():
            if key == "ee_pos":
                self._latest_obs["ee_pos_x"] = float(value[0])
                self._latest_obs["ee_pos_y"] = float(value[1])
                self._latest_obs["ee_pos_z"] = float(value[2])
            elif key == "ee_quat":
                self._latest_obs["ee_quat_x"] = float(value[0])
                self._latest_obs["ee_quat_y"] = float(value[1])
                self._latest_obs["ee_quat_z"] = float(value[2])
                self._latest_obs["ee_quat_w"] = float(value[3])
            # Standard joints
            else:
                self._latest_obs[f"{key}.pos"] = value

    def _on_control_request(self, sim_time):
        return self._target_action

    @property
    def observation_features(self) -> dict:
        obs = {
            "shoulder_pan.pos": float, "shoulder_lift.pos": float,
            "elbow_flex.pos": float, "wrist_flex.pos": float,
            "wrist_roll.pos": float, "gripper.pos": float,
            "realsense": (480, 640, 3),
        }
        
        if getattr(self.config, 'enable_ee_pose', True):
            obs.update({
                "ee_pos_x": float, "ee_pos_y": float, "ee_pos_z": float,
                "ee_quat_x": float, "ee_quat_y": float, "ee_quat_z": float, "ee_quat_w": float,
            })

        # Register depth feature
        if self.config.enable_depth:
            obs["realsense_depth"] = (480, 640, 3)
            obs["realsense_depth_vis"] = (480, 640, 3)  # The pretty image

        # Dynamically add the mono camera to the observation space if enabled
        if self.config.enable_wrist_cam:
            obs["wrist_cam"] = (480, 640, 3)
        return obs

    @property
    def action_features(self) -> dict:
        return {
            "shoulder_pan.pos": float, "shoulder_lift.pos": float,
            "elbow_flex.pos": float, "wrist_flex.pos": float,
            "wrist_roll.pos": float, "gripper.pos": float,
        }

    @property
    def is_connected(self) -> bool: return self._is_connected
    @property
    def is_calibrated(self) -> bool: return True
    def calibrate(self) -> None: pass
    def configure(self) -> None: pass

    def connect(self, calibrate: bool = True) -> None:
        self._sim_thread = threading.Thread(
            target=lambda: self.sim.run(headless=False), daemon=True)
        self._sim_thread.start()
        self._is_connected = True

        for key in self.action_features.keys():
            sim_key = key.replace(".pos", "")
            self._target_action[sim_key] = 0.0

        print("Waiting for MuJoCo to render the first frame...")
        
        # Build the list of required observation keys dynamically
        required_keys = ["realsense", "gripper.pos"]

        if getattr(self.config, 'enable_ee_pose', True):
            required_keys.extend([
                "ee_pos_x", "ee_pos_y", "ee_pos_z",
                "ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w"
            ])

        if self.config.enable_depth:
            required_keys.append("realsense_depth")
            required_keys.append("realsense_depth_vis")

        if self.config.enable_wrist_cam:
            required_keys.append("wrist_cam")

        while not all(k in self._latest_obs for k in required_keys):
            time.sleep(0.05)
        print("MuJoCo is ready!")

        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False
        if hasattr(self, 'sim'):
            self.sim.is_running = False

        if self._sim_thread is not None and self._sim_thread.is_alive():
            self._sim_thread.join()

    def restart_simulation(self) -> None:
        """Fully resets the MuJoCo environment, randomizes the scene, and opens a new window."""
        print("Shutting down old simulation...")
        self.disconnect()
        
        # Clear old observations so connect() waits properly for the new window to render
        self._latest_obs.clear()
        
        print("Building new randomized environment...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_xml_path = os.path.join(current_dir, self.config.xml_path)
        
        # Re-instantiate the simulation to reset time, MjData, and trigger scene randomization
        self.sim = SO101Simulation(
            xml_path=resolved_xml_path,
            camera_name=self.config.camera_name,
            render_fps=self.config.render_fps,
            enable_rgb=self.config.enable_rgb,
            enable_depth=self.config.enable_depth,
            show_cv2=self.config.show_cv2,
            enable_rerun=self.config.enable_rerun,
            rerun_log_meshes=self.config.rerun_log_meshes,
            rerun_log_tf=self.config.rerun_log_tf,
            rerun_depth_mode=self.config.rerun_depth_mode,
            rerun_log_rgb=self.config.rerun_log_rgb,
            depth_callback=self._on_depth_frame if self.config.enable_depth else None,
            wrist_callback=self._on_wrist_frame if self.config.enable_wrist_cam else None,
            rgb_callback=self._on_rgb_frame,
            joint_callback=self._on_joint_data,
            control_callback=self._on_control_request,
            scene_config=self.config
        )
        
        self.connect()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        return self._latest_obs.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        for key, val in action.items():
            if key.endswith(".pos"):
                sim_key = key.replace(".pos", "")
                self._target_action[sim_key] = val / self.scale
            else:
                self._target_action[key] = val
        return action
