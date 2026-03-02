import mujoco
import mujoco.viewer
import time
import numpy as np
import cv2
import uuid

# Only import rerun if available, to keep the module flexible
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False

# Only import pyroki if available
try:
    import pyroki as pk
    from robot_descriptions.loaders.yourdfpy import load_robot_description as load_urdf_description
    import pyroki_snippets as pks
    import viser
    from viser.extras import ViserUrdf
    PYROKI_AVAILABLE = True
except ImportError:
    PYROKI_AVAILABLE = False


class SO101Simulation:
    def __init__(
        self,
        xml_path: str,
        camera_name: str = "realsense_d435i",
        render_fps: int = 30,
        enable_rgb: bool = True,
        enable_depth: bool = True,
        show_cv2: bool = False,
        rgb_callback=None,
        depth_callback=None,
        joint_callback=None,  # <--- Added joint_callback,
        control_callback=None,  # <--- Added control_callback,  
        ik_callback=None,  # <--- NEW
        urdf_name: str = "so_arm101_description", # <--- NEW
        ik_target_link: str = "gripper", # <--- NEW
        ik_joint_mapping: dict = None, # <--- NEW,
        use_ik_web: bool = False, # <--- NEW: Enable Viser web interface
        # --- Rerun specific flags ---
        enable_rerun: bool = False,
        rerun_log_meshes: bool = True,
        rerun_log_tf: bool = True,
        rerun_depth_mode: str = "none",  # Options: "none", "depth", "pointcloud"
        rerun_log_rgb: bool = True,     # Toggle to hide/show the 2D RGB image in Rerun
    ):
        """
        Initializes the SO101 robot simulation with optional camera rendering, callbacks, and Rerun integration.
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.camera_name = camera_name
        self.render_fps = render_fps
        self.show_cv2 = show_cv2

        self.rgb_callback = rgb_callback
        self.depth_callback = depth_callback
        self.joint_callback = joint_callback  # <--- Store the callback
        self.control_callback = control_callback  # <--- Store the callback


        self.ik_callback = ik_callback
        self.use_ik_web = use_ik_web # <--- Store the flag
        self.urdf_name = urdf_name
        self.ik_target_link = ik_target_link
        self.ik_joint_mapping = ik_joint_mapping or {
            "1": "shoulder_pan", "2": "shoulder_lift", "3": "elbow_flex",
            "4": "wrist_flex", "5": "wrist_roll", "6": "gripper"
        }
        
        self._last_ik_pos = None
        self._last_ik_quat = None
        self.arm_base_id = 0

        if not PYROKI_AVAILABLE:
            print("Warning: ik_callback provided but Pyroki is not installed.")
        else:
            self._init_ik()

        # Rerun Configuration
        self.enable_rerun = enable_rerun and RERUN_AVAILABLE
        self.rerun_log_meshes = rerun_log_meshes
        self.rerun_log_tf = rerun_log_tf
        self.rerun_log_rgb = rerun_log_rgb

        # Validate depth mode
        valid_modes = ["none", "depth", "pointcloud"]
        if rerun_depth_mode not in valid_modes:
            print(
                f"Warning: Invalid rerun_depth_mode '{rerun_depth_mode}'. Defaulting to 'none'.")
            self.rerun_depth_mode = "none"
        else:
            self.rerun_depth_mode = rerun_depth_mode

        # Ensure rendering is enabled if required by callbacks, cv2, or rerun
        self.enable_rgb = enable_rgb or show_cv2 or (
            self.rerun_depth_mode == "pointcloud") or (self.enable_rerun and self.rerun_log_rgb)
        self.enable_depth = enable_depth or show_cv2 or (
            self.rerun_depth_mode in ["depth", "pointcloud"])

        if enable_rerun and not RERUN_AVAILABLE:
            print(
                "Warning: Rerun is enabled but not installed. Run 'pip install rerun-sdk'.")

        # Internal IDs for snapping
        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "d435i")
        self.mount_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "air_camera_mount")
        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        # Initial snap
        self._snap_camera()
        mujoco.mj_forward(self.model, self.data)

        # Setup Renderer
        self.width, self.height = 640, 480
        if self.enable_rgb or self.enable_depth:
            self.renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width)

        # Pre-calculate point cloud grid for efficiency
        if self.rerun_depth_mode == "pointcloud":
            self.fx = 400.0
            self.fy = 400.0
            self.cx = self.width / 2.0
            self.cy = self.height / 2.0
            u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
            self.u_flat = u.flatten()
            self.v_flat = v.flatten()

        # Initialize Rerun if enabled
        if self.enable_rerun:
            self._init_rerun()

    def _snap_camera(self):
        self.model.body_pos[self.cam_id] = self.model.body_pos[self.mount_id]
        self.model.body_quat[self.cam_id] = self.model.body_quat[self.mount_id]

    def _init_rerun(self):
        rand_id = str(uuid.uuid4())
        rr.init(rand_id, recording_id=rand_id, spawn=True)

        # Base camera node
        rr.log(f"world/{self.camera_name}", static=True)

        # Pinhole and ViewCoordinates
        rr.log(
            f"world/{self.camera_name}/optical",
            rr.Pinhole(resolution=[self.width, self.height], focal_length=400),
            rr.ViewCoordinates.RUB,
            static=True,
        )

        if self.rerun_log_meshes:
            self._log_static_meshes()

    def _log_static_meshes(self):
        for geom_id in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            if not body_name or body_name in {"box"}:
                continue

            if (self.model.geom_group[geom_id] > 2 or self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH):
                continue

            geom_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            entity_path = f"world/tf/{body_name}/{geom_name}"

            local_pos = self.model.geom_pos[geom_id]
            local_quat_wxyz = self.model.geom_quat[geom_id]
            local_quat_xyzw = [
                local_quat_wxyz[1], local_quat_wxyz[2], local_quat_wxyz[3], local_quat_wxyz[0]]

            rr.log(
                entity_path,
                rr.Transform3D(translation=local_pos,
                               rotation=rr.Quaternion(xyzw=local_quat_xyzw)),
                static=True,
            )

            mesh_id = self.model.geom_dataid[geom_id]
            if mesh_id == -1:
                continue

            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            vertices = self.model.mesh_vert[vert_adr: vert_adr + vert_num]

            face_adr = self.model.mesh_faceadr[mesh_id]
            face_num = self.model.mesh_facenum[mesh_id]
            faces = self.model.mesh_face[face_adr: face_adr + face_num]

            mat_id = self.model.geom_matid[geom_id]
            if mat_id != -1:
                rgba = self.model.mat_rgba[mat_id]
            else:
                rgba = self.model.geom_rgba[geom_id]

            color = (np.array([rgba[0], rgba[1], rgba[2],
                     rgba[3]]) * 255).astype(np.uint8)
            vertex_colors = np.tile(color, (vert_num, 1))

            rr.log(
                f"{entity_path}/mesh",
                rr.Mesh3D(vertex_positions=vertices,
                          triangle_indices=faces, vertex_colors=vertex_colors),
                static=True,
            )

    def _update_rerun_dynamic(self, rgb_image, raw_depth):
        IGNORED_BODIES = {"box"}

        # 1. Transform Tree
        for i in range(1, self.model.nbody):
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not body_name or body_name in IGNORED_BODIES:
                continue

            pos = self.data.xpos[i]
            quat_wxyz = self.data.xquat[i]
            quat_xyzw = [quat_wxyz[1], quat_wxyz[2],
                         quat_wxyz[3], quat_wxyz[0]]

            if self.rerun_log_tf:
                # By adding axis_length, Rerun will render the RGB gizmo automatically
                rr.log(
                    f"world/tf/{body_name}", 
                    rr.Transform3D(
                        translation=pos, 
                        rotation=rr.Quaternion(xyzw=quat_xyzw),
                        axis_length=0.05  # <--- Add this line to show the gizmo
                    )
                )
                parent_id = self.model.body_parentid[i]
                if parent_id != 0:
                    parent_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
                    if parent_name and parent_name not in IGNORED_BODIES:
                        parent_pos = self.data.xpos[parent_id]
                        rr.log(f"world/tf_skeleton/{parent_name}_to_{body_name}", rr.Arrows3D(origins=[
                               parent_pos], vectors=[pos - parent_pos], colors=[[150, 150, 150]], radii=0.002))
            else:
                rr.log(f"world/tf/{body_name}", rr.Transform3D(translation=pos,
                       rotation=rr.Quaternion(xyzw=quat_xyzw)))

        if self.rerun_log_tf and self.base_id != -1 and self.cam_id != -1:
            base_pos = self.data.xpos[self.base_id]
            cam_body_pos = self.data.xpos[self.cam_id]
            rr.log("world/visuals/base_to_camera", rr.Arrows3D(origins=[base_pos], vectors=[
                   cam_body_pos - base_pos], colors=[[255, 0, 0]], radii=0.005))
            
        # --- NEW: Log the IK Target to Rerun (Transformed to Global) ---
        if self.rerun_log_tf and self._last_ik_pos is not None:
            # Get the ground-truth global position and rotation matrix of the arm's base
            base_pos = self.data.xpos[self.arm_base_id]
            base_quat = self.data.xquat[self.arm_base_id]

            # Convert local Viser IK target to MuJoCo global coordinates
            global_ik_pos = [-self._last_ik_pos[1], self._last_ik_pos[0], self._last_ik_pos[2]]
            
            # --- NEW: Apply the +90 degree Z-rotation to the orientation ---
            # MuJoCo format is [w, x, y, z]. 
            # 90 deg around Z is [cos(45), 0, 0, sin(45)]
            z_90_quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])
            
            # First, rotate the Viser quaternion by our 90-degree offset
            corrected_viser_quat = np.zeros(4)
            mujoco.mju_mulQuat(corrected_viser_quat, z_90_quat, self._last_ik_quat)

            # Then, multiply it with the base_quat as you were doing before
            global_ik_quat = np.zeros(4)
            mujoco.mju_mulQuat(global_ik_quat, base_quat, corrected_viser_quat)
            # ----------------------------------------------------------------

            # Convert MuJoCo wxyz to Rerun xyzw
            ik_quat_xyzw = [
                global_ik_quat[1], global_ik_quat[2], 
                global_ik_quat[3], global_ik_quat[0]
            ]
            
            rr.log(
                "world/tf/ik_target", 
                rr.Transform3D(
                    translation=global_ik_pos, 
                    rotation=rr.Quaternion(xyzw=ik_quat_xyzw),
                    axis_length=0.1  # Make the target gizmo slightly larger/visible
                )
            )
        # ---------------------------------------
        
        # 2. Log RGB (Only if explicitly enabled for Rerun)
        if self.rerun_log_rgb and self.enable_rgb and rgb_image is not None:
            rr.log(f"world/{self.camera_name}/optical/rgb",
                   rr.Image(rgb_image))

        # 3. Log Depth (Mutually Exclusive Modes)
        if raw_depth is not None:
            if self.rerun_depth_mode == "depth":
                rr.log(f"world/{self.camera_name}/optical/depth",
                       rr.DepthImage(raw_depth, meter=1.0))

            elif self.rerun_depth_mode == "pointcloud" and rgb_image is not None:
                z = raw_depth.flatten()
                colors = rgb_image.reshape(-1, 3)

                valid = (z > 0.05) & (z < 5.0)
                z_valid = z[valid]
                u_valid = self.u_flat[valid]
                v_valid = self.v_flat[valid]
                colors_valid = colors[valid]

                x_valid = (u_valid - self.cx) * z_valid / self.fx
                y_valid = (v_valid - self.cy) * z_valid / self.fy

                positions = np.vstack((x_valid, -y_valid, -z_valid)).T
                rr.log(f"world/{self.camera_name}/pointcloud",
                       rr.Points3D(positions, colors=colors_valid))

        # 4. Update Main Camera Extrinsic Transform
        actual_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if actual_cam_id != -1:
            cam_pos = self.data.cam_xpos[actual_cam_id]
            cam_mat = self.data.cam_xmat[actual_cam_id]
            cam_quat_wxyz = np.zeros(4)
            mujoco.mju_mat2Quat(cam_quat_wxyz, cam_mat)
            cam_quat_xyzw = [cam_quat_wxyz[1], cam_quat_wxyz[2],
                             cam_quat_wxyz[3], cam_quat_wxyz[0]]

            rr.log(
                f"world/{self.camera_name}",
                rr.Transform3D(translation=cam_pos,
                               rotation=rr.Quaternion(xyzw=cam_quat_xyzw)),
            )

    # --- NEW METHOD TO EXTRACT JOINTS ---
    def _process_joints(self):
        if self.joint_callback is None:
            return

        joint_data = {}
        for i in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name:
                # Get the index in the qpos array for this joint
                qpos_idx = self.model.jnt_qposadr[i]

                # Assuming 1-DOF joints (hinge or prismatic) for an arm
                # We grab the scalar angle/position directly
                joint_data[jnt_name] = self.data.qpos[qpos_idx]

        self.joint_callback(joint_data)

    def apply_commands(self, commands):
        if not commands:
            return

        for name, value in commands.items():
            # Find the ID of the actuator by its name
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

            if act_id != -1:
                # Apply the target value to the actuator's control array
                self.data.ctrl[act_id] = value
            else:
                # Optional: print a warning if the name doesn't match your XML
                # print(f"Warning: Actuator '{name}' not found.")
                pass

    def _init_ik(self):
        self.urdf = load_urdf_description(self.urdf_name)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.urdf_joints = [j.name for j in self.urdf.actuated_joints]
        self.map_indices = []
        
        for u_idx, u_name in enumerate(self.urdf_joints):
            if u_name in self.ik_joint_mapping:
                m_name = self.ik_joint_mapping[u_name]
                m_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, m_name)
                if m_id != -1:
                    self.map_indices.append((u_idx, m_id))

        # Setup Viser Web Server if enabled
        if self.use_ik_web:
            self.viser_server = viser.ViserServer()
            self.viser_server.scene.add_grid("/ground", width=2, height=2)
            self.urdf_vis = ViserUrdf(self.viser_server, self.urdf, root_node_name="/ghost_robot")
            self.ik_web_target = self.viser_server.scene.add_transform_controls(
                "/ik_target", scale=0.1, position=(0.3, 0.0, 0.2), wxyz=(1, 0, 0, 0)
            )
        
        # Dynamically find the true arm base ID for coordinate math
        first_actuator = self.ik_joint_mapping.get("1")
        if first_actuator:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, first_actuator)
            if act_id != -1:
                # Get joint -> body -> parent body (the true base)
                jnt_id = self.model.actuator_trnid[act_id, 0]
                body_id = self.model.jnt_bodyid[jnt_id]
                self.arm_base_id = self.model.body_parentid[body_id]

    def apply_ik(self, ik_command):
        if not ik_command:
            return

        target_pos = ik_command.get("pos")
        target_gripper = ik_command.get("gripper", 0.0)

        if target_pos is None:
            return

        # Check if quaternion is explicitly provided (for Web view)
        if "quat" in ik_command:
            target_quat = ik_command["quat"]
        else:
            target_rpy = ik_command.get("rpy", [0.0, 0.0, 0.0])
            target_quat = np.zeros(4)
            mujoco.mju_euler2Quat(target_quat, target_rpy, "xyz")
        
        # Save target for Rerun visualizer
        self._last_ik_pos = target_pos
        self._last_ik_quat = target_quat

        # Solve IK
        q_sol = pks.solve_ik(
            robot=self.robot,
            target_link_name=self.ik_target_link,
            target_position=target_pos,
            target_wxyz=target_quat,
        )

        if q_sol is not None:
            # Update web visualizer ghost if enabled
            if self.use_ik_web:
                self.urdf_vis.update_cfg(q_sol)
                
            for u_idx, m_idx in self.map_indices:
                u_name = self.urdf_joints[u_idx]
                if u_name == "6": 
                    self.data.ctrl[m_idx] = target_gripper 
                else:
                    self.data.ctrl[m_idx] = q_sol[u_idx]

    def _process_cameras(self):
        rgb_image = None
        bgr_image = None
        raw_depth = None
        depth_colormap = None

        if self.enable_rgb:
            self.renderer.update_scene(self.data, camera=self.camera_name)
            self.renderer.disable_depth_rendering()
            rgb_image = self.renderer.render()
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            if self.rgb_callback:
                self.rgb_callback(bgr_image)

        if self.enable_depth:
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=self.camera_name)
            raw_depth = self.renderer.render()

            if self.show_cv2 or self.depth_callback:
                max_depth = 3.0
                depth_visual = raw_depth.copy()
                bg_mask = (depth_visual == 0.0) | np.isinf(
                    depth_visual) | np.isnan(depth_visual)
                depth_visual[bg_mask] = max_depth

                depth_normalized = np.clip(
                    depth_visual, 0, max_depth) / max_depth
                depth_8bit = (depth_normalized * 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(
                    255 - depth_8bit, cv2.COLORMAP_JET)

                if self.depth_callback:
                    self.depth_callback(raw_depth, depth_colormap)

        if self.enable_rerun:
            self._update_rerun_dynamic(rgb_image, raw_depth)

        if self.show_cv2:
            if bgr_image is not None:
                cv2.imshow("RGB Camera", bgr_image)
            if depth_colormap is not None:
                cv2.imshow("Depth Camera", depth_colormap)
            cv2.waitKey(1)

    def run(self):
        sim_start_time = time.time()
        render_interval = 1.0 / self.render_fps
        last_render_time = time.time()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                real_elapsed_time = time.time() - sim_start_time

                while self.data.time < real_elapsed_time:
                    # --- NEW: Get and apply commands before stepping ---
                    if self.control_callback:
                        commands = self.control_callback(self.data.time)
                        if isinstance(commands, dict):
                            self.apply_commands(commands)
                    # --- NEW: IK Coordinate Control ---
                    if PYROKI_AVAILABLE:
                        ik_command = None
                        
                        # Prioritize Web view if enabled
                        if self.use_ik_web:
                            ik_command = {
                                "pos": np.array(self.ik_web_target.position),
                                "quat": np.array(self.ik_web_target.wxyz),
                                "gripper": 0.0 # Gripper logic handled separately or keep default
                            }
                        elif self.ik_callback:
                            ik_command = self.ik_callback(self.data.time)
                            
                        if isinstance(ik_command, dict):
                            self.apply_ik(ik_command)
                    # ---------------------------------------------------
                    mujoco.mj_step(self.model, self.data)
                    self._snap_camera()
                    mujoco.mj_forward(self.model, self.data)

                viewer.sync()

                current_time = time.time()
                # Run visual and joint processing synced at the render_fps rate
                if current_time - last_render_time >= render_interval:
                    if self.enable_rgb or self.enable_depth:
                        self._process_cameras()

                    self._process_joints()  # <--- Added joint processing hook
                    last_render_time = current_time

        if self.show_cv2:
            cv2.destroyAllWindows()
