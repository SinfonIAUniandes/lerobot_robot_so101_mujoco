from dataclasses import dataclass
from lerobot.robots import RobotConfig

@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class So101MujocoRobotConfig(RobotConfig):
    # Core simulation parameters
    xml_path: str = "./robotstudio_so101/so101_camera_mount.xml"
    camera_name: str = "realsense_d435i"
    render_fps: int = 30
    
    # Rendering toggles
    enable_rgb: bool = True
    enable_depth: bool = False
    show_cv2: bool = False
    
    # Rerun.io telemetry integration
    enable_rerun: bool = False
    rerun_log_meshes: bool = True
    rerun_log_tf: bool = True
    rerun_depth_mode: str = "none"  # "none", "depth", or "pointcloud"
    rerun_log_rgb: bool = True