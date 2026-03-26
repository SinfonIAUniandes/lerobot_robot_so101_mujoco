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
    
    # Scene randomization parameters
    randomize_scene: bool = False
    
    # Black Box (Base values + allowed variance)
    box_pos_base: tuple = (0.35, 0.0, 0.03)
    box_pos_delta: tuple = (0.05, 0.10, 0.0) # Varies X by ±0.05, Y by ±0.10
    box_size_base: tuple = (0.02, 0.02, 0.03)
    box_size_delta: tuple = (0.005, 0.005, 0.005)
    box_color_base: tuple = (0.1, 0.1, 0.1, 1.0) # Base black (r,g,b,alpha)
    box_color_delta: tuple = (0.1, 0.1, 0.1, 0.0) # Can drift slightly lighter
    
    # Red Tray
    tray_pos_base: tuple = (0.35, 0.2, 0.01)
    tray_pos_delta: tuple = (0.10, 0.05, 0.0)
    tray_size_base: tuple = (0.08, 0.08, 0.01)
    tray_size_delta: tuple = (0.02, 0.02, 0.0)
    tray_color_base: tuple = (0.8, 0.1, 0.1, 1.0) # Base red
    tray_color_delta: tuple = (0.2, 0.1, 0.1, 0.0) # Can drift slightly in hue/brightness
    
    # Camera Mount
    camera_pos_base: tuple = (0.2, -0.4, 0.5)
    camera_pos_delta: tuple = (0.0, 0.0, 0.0)
    camera_euler_base: tuple = (0.785398, 3.14159, 0.0) # Base orientation (45° down, facing forward)
    camera_euler_delta: tuple = (0.0, 0.0, 0.0)