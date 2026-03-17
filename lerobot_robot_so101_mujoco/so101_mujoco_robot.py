import os
import threading
import numpy as np
from typing import Any
from lerobot.robots.robot import Robot
import time

from .so101_sim import SO101Simulation
from .config_so101_mujoco_robot import So101MujocoRobotConfig

import mujoco

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

        self.cameras = {"camera": None}

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
            rgb_callback=self._on_rgb_frame,
            joint_callback=self._on_joint_data,
            control_callback=self._on_control_request,
            scene_config=self.config
        )

    def _on_rgb_frame(self, bgr_image):
        import cv2
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self._latest_obs["camera"] = rgb_image

    def _on_joint_data(self, joint_data):
        for joint_name, pos in joint_data.items():
            self._latest_obs[f"{joint_name}.pos"] = pos

        # --- NEW: Proprioceptive Force Calculation ---
        try:
            ee_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
            if ee_id != -1:
                # 1. Get the 3D Translation Jacobian for the gripper
                # nv is the number of degrees of freedom in the MuJoCo model
                jacp = np.zeros((3, self.sim.model.nv))
                mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, None, ee_id)

                # 2. Get the actual torque effort from the motors
                # qfrc_actuator contains the torques currently applied by your PD controllers
                tau = self.sim.data.qfrc_actuator.copy()

                # 3. Calculate Cartesian Force: F = (J^T)^+ * tau
                # We use pseudo-inverse (pinv) so the math doesn't explode if the arm fully extends (singularity)
                J_trans_pinv = np.linalg.pinv(jacp.T)
                cartesian_force = J_trans_pinv @ tau

                self._latest_obs["motor_force"] = cartesian_force
        except Exception as e:
            pass

    def _on_control_request(self, sim_time):
        return self._target_action

    @property
    def observation_features(self) -> dict:
        return {
            "shoulder_pan.pos": float, "shoulder_lift.pos": float,
            "elbow_flex.pos": float, "wrist_flex.pos": float,
            "wrist_roll.pos": float, "gripper.pos": float, 
            "camera": (480, 640, 3),
            "motor_force": (3,)
        }
        
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
        self._sim_thread = threading.Thread(target=lambda: self.sim.run(headless=False), daemon=True)
        self._sim_thread.start()
        self._is_connected = True

        for key in self.action_features.keys():
            sim_key = key.replace(".pos", "")
            self._target_action[sim_key] = 0.0
            
        print("Waiting for MuJoCo to render the first frame...")
        while "camera" not in self._latest_obs or "gripper.pos" not in self._latest_obs:
            time.sleep(0.05)
        print("MuJoCo is ready!")

        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False
        if hasattr(self, 'sim'):
            self.sim.is_running = False

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