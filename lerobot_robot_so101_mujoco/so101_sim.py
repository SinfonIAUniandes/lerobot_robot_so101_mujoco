import mujoco
import mujoco.viewer
import time
import numpy as np
import cv2
import uuid

# Only import rerun if available
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


class SO101Simulation:
    def __init__(
        self,
        xml_path: str,
        camera_name: str = "realsense_d435i",
        render_fps: int = 30,
        enable_rgb: bool = True,
        enable_depth: bool = False,
        show_cv2: bool = False,
        rgb_callback=None,
        depth_callback=None,
        wrist_callback=None,
        joint_callback=None,
        control_callback=None,
        enable_rerun: bool = False,
        rerun_log_meshes: bool = True,
        rerun_log_tf: bool = True,
        rerun_depth_mode: str = "none",  # "none", "depth", "pointcloud"
        rerun_log_rgb: bool = True,
        scene_config=None
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.scene_config = scene_config

        # 1. CREATE DATA FIRST so we can modify the dynamic qpos state
        self.data = mujoco.MjData(self.model)

        # Always run setup so we use the config's base values
        if self.scene_config:
            self._setup_scene()

        self.camera_name = camera_name
        self.render_fps = render_fps
        self.show_cv2 = show_cv2

        self.rgb_callback = rgb_callback
        self.depth_callback = depth_callback
        self.wrist_callback = wrist_callback
        self.joint_callback = joint_callback
        self.control_callback = control_callback
        
        # Fetch wrist cam settings from config
        self.enable_wrist_cam = getattr(self.scene_config, 'enable_wrist_cam', False)
        self.wrist_camera_name = getattr(self.scene_config, 'wrist_camera_name', 'wrist_cam')

        # Rerun Configuration
        self.enable_rerun = enable_rerun and RERUN_AVAILABLE
        self.rerun_log_meshes = rerun_log_meshes
        self.rerun_log_tf = rerun_log_tf
        self.rerun_log_rgb = rerun_log_rgb

        valid_modes = ["none", "depth", "pointcloud"]
        if rerun_depth_mode not in valid_modes:
            print(
                f"Warning: Invalid rerun_depth_mode '{rerun_depth_mode}'. Defaulting to 'none'.")
            self.rerun_depth_mode = "none"
        else:
            self.rerun_depth_mode = rerun_depth_mode

        self.enable_rgb = enable_rgb or show_cv2 or (
            self.rerun_depth_mode == "pointcloud") or (self.enable_rerun and self.rerun_log_rgb)
        self.enable_depth = enable_depth or show_cv2 or (
            self.rerun_depth_mode in ["depth", "pointcloud"])

        if enable_rerun and not RERUN_AVAILABLE:
            print(
                "Warning: Rerun is enabled but not installed. Run 'pip install rerun-sdk'.")

        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "d435i")
        self.mount_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "air_camera_mount")
        self.base_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        self._snap_camera()
        mujoco.mj_forward(self.model, self.data)

        self.width, self.height = 640, 480
        self.renderer = None
        self.wrist_renderer = None

        if self.rerun_depth_mode == "pointcloud":
            self.fx = 400.0
            self.fy = 400.0
            self.cx = self.width / 2.0
            self.cy = self.height / 2.0
            u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
            self.u_flat = u.flatten()
            self.v_flat = v.flatten()

        if self.enable_rerun:
            self._init_rerun()

    def _setup_scene(self):
        is_rand = getattr(self.scene_config, 'randomize_scene', False)

        def get_val(base, delta):
            base_arr = np.array(base, dtype=np.float32)
            if not is_rand:
                return base_arr
            delta_arr = np.array(delta, dtype=np.float32)
            # Generate random noise between -delta and +delta
            noise = np.random.uniform(-delta_arr, delta_arr)
            return base_arr + noise

        # 1. Setup Box
        box_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")

        if box_body_id != -1:
            new_pos = get_val(self.scene_config.box_pos_base,
                              self.scene_config.box_pos_delta)

            # Check if the body has a joint (like our freejoint)
            jnt_adr = self.model.body_jntadr[box_body_id]
            if jnt_adr != -1:
                # Modifying qpos moves the free joint. Updating qpos0 ensures resets work.
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                self.data.qpos[qpos_adr: qpos_adr + 3] = new_pos
                self.model.qpos0[qpos_adr: qpos_adr + 3] = new_pos
            else:
                # Fallback for static bodies (no freejoint)
                self.model.body_pos[box_body_id] = new_pos

        if box_geom_id != -1:
            self.model.geom_size[box_geom_id] = get_val(
                self.scene_config.box_size_base, self.scene_config.box_size_delta)
            # Clip colors to ensure they stay between 0.0 and 1.0
            self.model.geom_rgba[box_geom_id] = np.clip(get_val(
                self.scene_config.box_color_base, self.scene_config.box_color_delta), 0.0, 1.0)

        # 2. Setup Tray
        tray_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tray")
        tray_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "tray_geom")
        if tray_body_id != -1:
            self.model.body_pos[tray_body_id] = get_val(
                self.scene_config.tray_pos_base, self.scene_config.tray_pos_delta)
        if tray_geom_id != -1:
            self.model.geom_size[tray_geom_id] = get_val(
                self.scene_config.tray_size_base, self.scene_config.tray_size_delta)
            self.model.geom_rgba[tray_geom_id] = np.clip(get_val(
                self.scene_config.tray_color_base, self.scene_config.tray_color_delta), 0.0, 1.0)

        # 3. Setup Camera
        cam_mount_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "air_camera_mount")
        if cam_mount_id != -1:
            self.model.body_pos[cam_mount_id] = get_val(
                self.scene_config.camera_pos_base, self.scene_config.camera_pos_delta)

            # Apply orientation via euler (convert to quat and set body_quat)
            if hasattr(self.scene_config, "camera_euler_base") and hasattr(self.scene_config, "camera_euler_delta"):
                euler_target = get_val(
                    self.scene_config.camera_euler_base, self.scene_config.camera_euler_delta)
                quat_target = np.zeros(4)
                mujoco.mju_euler2Quat(quat_target, euler_target, "XYZ")
                self.model.body_quat[cam_mount_id] = quat_target

    def _snap_camera(self):
        self.model.body_pos[self.cam_id] = self.model.body_pos[self.mount_id]
        self.model.body_quat[self.cam_id] = self.model.body_quat[self.mount_id]

    def _init_rerun(self):
        rand_id = str(uuid.uuid4())
        rr.init(rand_id, recording_id=rand_id, spawn=True)

        rr.log(f"world/{self.camera_name}", static=True)
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

            rr.log(entity_path, rr.Transform3D(translation=local_pos,
                   rotation=rr.Quaternion(xyzw=local_quat_xyzw)), static=True)

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

            rr.log(f"{entity_path}/mesh", rr.Mesh3D(vertex_positions=vertices,
                   triangle_indices=faces, vertex_colors=vertex_colors), static=True)

    def _update_rerun_dynamic(self, rgb_image, raw_depth):
        IGNORED_BODIES = {"box"}

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
                rr.log(f"world/tf/{body_name}", rr.Transform3D(translation=pos,
                       rotation=rr.Quaternion(xyzw=quat_xyzw), axis_length=0.05))
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

        if self.rerun_log_rgb and self.enable_rgb and rgb_image is not None:
            rr.log(f"world/{self.camera_name}/optical/rgb",
                   rr.Image(rgb_image))

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

        actual_cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if actual_cam_id != -1:
            cam_pos = self.data.cam_xpos[actual_cam_id]
            cam_mat = self.data.cam_xmat[actual_cam_id]
            cam_quat_wxyz = np.zeros(4)
            mujoco.mju_mat2Quat(cam_quat_wxyz, cam_mat)
            cam_quat_xyzw = [cam_quat_wxyz[1], cam_quat_wxyz[2],
                             cam_quat_wxyz[3], cam_quat_wxyz[0]]

            rr.log(f"world/{self.camera_name}", rr.Transform3D(
                translation=cam_pos, rotation=rr.Quaternion(xyzw=cam_quat_xyzw)))

    def _process_joints(self):
        if self.joint_callback is None:
            return

        joint_data = {}
        for i in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name:
                qpos_idx = self.model.jnt_qposadr[i]
                joint_data[jnt_name] = self.data.qpos[qpos_idx]

        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "baseframe")

        if ee_id != -1 and base_id != -1:
            # 1. Extract World Space Data (and reshape 1D matrices to 3x3)
            pos_ee_world = self.data.site_xpos[ee_id]
            mat_ee_world = self.data.site_xmat[ee_id].reshape(3, 3)
            
            pos_base_world = self.data.site_xpos[base_id]
            mat_base_world = self.data.site_xmat[base_id].reshape(3, 3)

            # 2. Compute Relative Position (Base-centric coordinate system)
            # Math: Multiply transposed base rotation matrix by positional difference
            pos_rel = mat_base_world.T @ (pos_ee_world - pos_base_world)

            # 3. Compute Relative Rotation (Matrix to Quaternion)
            mat_rel = mat_base_world.T @ mat_ee_world
            quat_rel_wxyz = np.zeros(4)
            # MuJoCo natively outputs (w, x, y, z) quaternions
            mujoco.mju_mat2Quat(quat_rel_wxyz, mat_rel.flatten())

            quat_rel_xyzw = np.array([
                quat_rel_wxyz[1],  # x
                quat_rel_wxyz[2],  # y
                quat_rel_wxyz[3],  # z
                quat_rel_wxyz[0]   # w
            ], dtype=np.float32)

            # Cast to float32 for deep learning model compatibility
            joint_data["ee_pos"] = pos_rel.astype(np.float32)
            joint_data["ee_quat"] = quat_rel_xyzw

        self.joint_callback(joint_data)

    def apply_commands(self, commands):
        if not commands:
            return

        for name, value in commands.items():
            act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1:
                self.data.ctrl[act_id] = value

    def _process_cameras(self):
        if self.renderer is None and (self.enable_rgb or self.enable_depth):
            self.renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width)

        # Initialize secondary renderer for wrist cam
        if self.enable_wrist_cam and self.wrist_renderer is None:
            self.wrist_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

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

        if self.enable_wrist_cam:
            self.wrist_renderer.update_scene(self.data, camera=self.wrist_camera_name)
            wrist_rgb = self.wrist_renderer.render()

            if self.wrist_callback:
                self.wrist_callback(wrist_rgb)

        if self.enable_rerun:
            self._update_rerun_dynamic(rgb_image, raw_depth)

        if self.show_cv2:
            if bgr_image is not None:
                cv2.imshow("RGB Camera", bgr_image)
            if depth_colormap is not None:
                cv2.imshow("Depth Camera", depth_colormap)
            if self.enable_wrist_cam and 'wrist_mono' in locals():
                cv2.imshow("Arducam Wrist (Mono)", wrist_mono)
            cv2.waitKey(1)

    def run(self, headless=False):
        sim_start_time = time.time()
        render_interval = 1.0 / self.render_fps
        last_render_time = time.time()
        self.is_running = True

        viewer = mujoco.viewer.launch_passive(
            self.model, self.data) if not headless else None

        try:
            while self.is_running:
                if viewer and not viewer.is_running():
                    break

                real_elapsed_time = time.time() - sim_start_time

                while self.data.time < real_elapsed_time:

                    if not self.is_running:
                        break

                    if self.control_callback:
                        commands = self.control_callback(self.data.time)
                        if isinstance(commands, dict):
                            self.apply_commands(commands)

                    mujoco.mj_step(self.model, self.data)
                    self._snap_camera()
                    mujoco.mj_forward(self.model, self.data)

                if viewer:
                    viewer.sync()
                else:
                    time.sleep(0.001)

                current_time = time.time()
                if current_time - last_render_time >= render_interval:
                    if self.enable_rgb or self.enable_depth:
                        self._process_cameras()

                    self._process_joints()
                    last_render_time = current_time
        finally:
            if viewer:
                viewer.close()
            if self.show_cv2:
                cv2.destroyAllWindows()
            if hasattr(self, 'renderer') and self.renderer is not None:
                self.renderer.close()
            if hasattr(self, 'wrist_renderer') and self.wrist_renderer is not None:
                self.wrist_renderer.close() # Clean up the second renderer
