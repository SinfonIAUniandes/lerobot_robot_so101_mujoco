from dataclasses import dataclass
from lerobot.robots import RobotConfig

@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class So101MujocoRobotConfig(RobotConfig):
    # Default paths mapped to your simulation's structure
    xml_path: str = "./robotstudio_so101/so101_camera_mount.xml"
    urdf_name: str = "so_arm101_description"