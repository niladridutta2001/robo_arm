"""Minimal PyBullet scene for the Robot-A portion of the assignment."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from control import VisualServoController
from metrics import Metrics


ROOT = Path(__file__).resolve().parent
DT = 1.0 / 240.0
ARM_SPEED = 2.25  # rad/s; below the URDF limits of 3.15--3.2 rad/s
ARM_ACCELERATION = 60.0  # rad/s^2; shared with the standalone TOPP planner
ARM_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
PATTERN_CENTER = np.array((-0.18, 0.0, 1.05))
PAYLOAD_Z_COMPENSATION = 0.033  # m, measured gravity/compliance feed-forward


def smoothstep(u: float) -> float:
    u = np.clip(u, 0.0, 1.0)
    return float(u * u * (3.0 - 2.0 * u))


class Demo:
    def __init__(self, gui: bool = True, realtime: bool = True, method: str = "ik",
                 robot_a_motion: str = "baseline"):
        self.gui, self.realtime = gui, realtime
        self.method = method
        self.robot_a_motion = robot_a_motion
        suffix = "" if robot_a_motion == "baseline" else f"_{robot_a_motion}"
        self.run_dir = ROOT / "metrics" / "runs" / f"run_{method}{suffix}"
        self.artifact_suffix = self.run_dir.name.removeprefix("run_")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_log = []
        self.sim_steps = 0
        self.camera_rgb = None
        self.video_writer = None
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.setPhysicsEngineParameter(numSolverIterations=120)
        self._make_scene()

    def _make_scene(self) -> None:
        p.loadURDF("plane.urdf")
        self._box((-0.62, 0, 0.35), (0.65, 0.85, 0.70), (0.55, 0.38, 0.24, 1))
        self._box((0.72, 0, 0.35), (0.65, 0.85, 0.70), (0.55, 0.38, 0.24, 1))

        urdf = ROOT / "assets" / "ur5" / "urdf" / "ur5_robotiq_85.urdf"
        self.ur5 = p.loadURDF(str(urdf), (-0.72, 0, 0.70), useFixedBase=True)
        self.panda = p.loadURDF(
            "franka_panda/panda.urdf", (0.72, 0, 0.70),
            p.getQuaternionFromEuler((0, 0, math.pi)), useFixedBase=True,
        )
        panda_joints = {p.getJointInfo(self.panda, i)[1].decode(): i
                        for i in range(p.getNumJoints(self.panda))}
        self.camera_link = panda_joints["panda_hand_joint"]
        self.franka_controller = VisualServoController(
            self.panda, self.camera_link, method=self.method
        )
        self.sync_enabled = False

        self.joints = {p.getJointInfo(self.ur5, i)[1].decode(): i
                       for i in range(p.getNumJoints(self.ur5))}
        self.arm_joints = [self.joints[name] for name in ARM_NAMES]
        self.ee = self.joints["robotiq_85_base_joint"]
        self.finger_names = [
            "finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint",
            "right_outer_knuckle_joint", "right_inner_finger_joint",
            "right_inner_knuckle_joint",
        ]
        home = (0.0, -1.55, 1.65, -1.67, -1.57, 0.0)
        self.ur5_command = np.asarray(home, dtype=float)
        self.ur5_command_velocity = np.zeros(6)
        for j, q in zip(self.arm_joints, home):
            p.resetJointState(self.ur5, j, q)
            p.setJointMotorControl2(
                self.ur5, j, p.POSITION_CONTROL, targetPosition=q,
                force=p.getJointInfo(self.ur5, j)[10], maxVelocity=ARM_SPEED,
                positionGain=0.25,
            )

        cube_pos = (-0.48, -0.23, 0.765)
        if self.method == "kf_pose":
            self.cube = self._marked_cube(cube_pos)
        else:
            self.cube = self._box(cube_pos, (0.055, 0.055, 0.055),
                                  (0.9, 0.08, 0.08, 1), 0.08)
        self.open_gripper()
        self._set_panda_rest_pose()
        self.metrics = Metrics(self.ur5, self.arm_joints, self.panda, self.cube,
                               DT, control_method=self.method,
                               ur5_acceleration_limit=(ARM_ACCELERATION
                                                       if self.robot_a_motion == "topp"
                                                       else None),
                               robot_a_motion=self.robot_a_motion)

        if self.gui:
            p.resetDebugVisualizerCamera(2.15, 48, -25, (0, 0, 0.82))

    @staticmethod
    def _box(pos, size, color, mass=0.0):
        half = np.asarray(size) / 2.0
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        return p.createMultiBody(mass, collision, visual, basePosition=pos)

    @staticmethod
    def _marked_cube(pos):
        """Red cube with a textured ArUco marker on its +x face."""
        import cv2

        half = (0.0275, 0.0275, 0.0275)
        base_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=(0.9, 0.08, 0.08, 1)
        )
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        marker_visual = p.createVisualShape(
            p.GEOM_MESH, fileName=str(ROOT / "assets" / "aruco_marker.obj"),
            rgbaColor=(1, 1, 1, 1)
        )
        body = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=pos,
            linkMasses=[0.0],
            linkCollisionShapeIndices=[-1],
            linkVisualShapeIndices=[marker_visual],
            linkPositions=[(0.0282, 0, 0)],
            linkOrientations=[(0, 0, 0, 1)],
            linkInertialFramePositions=[(0, 0, 0)],
            linkInertialFrameOrientations=[(0, 0, 0, 1)],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[(0, 0, 0)],
        )
        texture_path = ROOT / "assets" / "aruco_23.png"
        if not texture_path.exists():
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            if hasattr(cv2.aruco, "generateImageMarker"):
                marker = cv2.aruco.generateImageMarker(dictionary, 23, 256)
            else:
                marker = np.zeros((256, 256), dtype=np.uint8)
                cv2.aruco.drawMarker(dictionary, 23, 256, marker, 1)
            texture = np.full((320, 320), 255, dtype=np.uint8)
            texture[32:288, 32:288] = marker
            if not cv2.imwrite(str(texture_path), texture):
                raise RuntimeError(f"Could not write ArUco texture {texture_path}")
        texture_id = p.loadTexture(str(texture_path))
        p.changeVisualShape(body, 0, textureUniqueId=texture_id)
        return body

    def _set_panda_rest_pose(self):
        rest = (0, -0.55, 0, -2.25, 0, 1.75, 0.8, 0.04, 0.04)
        for j, q in zip(range(9), rest):
            p.resetJointState(self.panda, j, q)
            if p.getJointInfo(self.panda, j)[2] != p.JOINT_FIXED:
                info = p.getJointInfo(self.panda, j)
                p.setJointMotorControl2(
                    self.panda, j, p.POSITION_CONTROL, targetPosition=q,
                    force=info[10], maxVelocity=min(1.0, info[11]),
                    positionGain=0.3,
                )

    def step(self, seconds: float):
        for _ in range(max(1, round(seconds / DT))):
            p.stepSimulation()
            self.metrics.physics_step(self.stage)
            self.sim_steps += 1
            if self.sim_steps % 8 == 0:  # 30 Hz camera
                self.render_franka_camera()
                if self.video_writer is not None:
                    self.capture_video_frame()
            if self.realtime:
                time.sleep(DT)

    def start_video(self, path):
        """Start deterministic 30 Hz scene recording with OpenCV."""
        import cv2

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (960, 540)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the MP4 video writer")
        self.video_writer = writer
        self.video_path = path

    def capture_video_frame(self):
        import cv2

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0, 0, 0.82), distance=2.15,
            yaw=48, pitch=-25, roll=0, upAxisIndex=2,
        )
        projection = p.computeProjectionMatrixFOV(
            fov=60, aspect=16 / 9, nearVal=0.05, farVal=5.0
        )
        frame = p.getCameraImage(
            960, 540, view, projection,
            renderer=(p.ER_BULLET_HARDWARE_OPENGL if self.gui
                      else p.ER_TINY_RENDERER),
        )
        rgba = np.asarray(frame[2], dtype=np.uint8).reshape(540, 960, 4)
        self.video_writer.write(cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR))

    def stop_video(self):
        if self.video_writer is None:
            return None
        self.video_writer.release()
        self.video_writer = None
        if not self.video_path.exists() or self.video_path.stat().st_size == 0:
            raise RuntimeError(f"Video writer produced no file: {self.video_path}")
        return self.video_path

    def render_franka_camera(self):
        """Render Robot B's sensor; later control will receive only these arrays."""
        if not p.isConnected():
            return False
        camera_pos, hand_orn = p.getLinkState(self.panda, self.camera_link)[4:6]
        if not hasattr(self, "camera_forward_local"):
            forward = PATTERN_CENTER - np.asarray(camera_pos)
            distance = np.linalg.norm(forward)
            if distance < 1e-6:
                return False
            forward /= distance
            _, inverse_orn = p.invertTransform((0, 0, 0), hand_orn)
            self.camera_forward_local = p.rotateVector(inverse_orn, forward)
            self.camera_up_local = p.rotateVector(inverse_orn, (0, 0, 1))
        forward = p.rotateVector(hand_orn, self.camera_forward_local)
        up = p.rotateVector(hand_orn, self.camera_up_local)
        target = np.asarray(camera_pos) + np.asarray(forward)
        view = p.computeViewMatrix(camera_pos, target, up)
        projection = p.computeProjectionMatrixFOV(60, 4 / 3, 0.05, 3.0)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        frame = p.getCameraImage(320, 240, view, projection, renderer=renderer)
        if frame is None:
            return False
        _, _, rgba, depth, segmentation = frame
        self.camera_rgb = np.asarray(rgba, dtype=np.uint8).reshape(240, 320, 4)[..., :3]
        self.camera_depth = np.asarray(depth).reshape(240, 320)
        self.camera_segmentation = np.asarray(segmentation).reshape(240, 320)
        if self.sync_enabled:
            detected = self.franka_controller.update(
                self.camera_rgb, self.camera_depth, forward, up
            )
            self.metrics.camera_sample(
                self.franka_controller, camera_pos, forward, up, detected,
                self.sim_steps * DT,
            )
        return True

    def save_camera_view(self, path=None):
        path = path or self.run_dir / f"camera_view_{self.artifact_suffix}.png"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.camera_rgb is None:
            if not self.render_franka_camera():
                print("Warning: Franka camera did not return an image")
                return
        from matplotlib import image as mpimg
        mpimg.imsave(path, self.camera_rgb)
        return path

    def set_gripper(self, opening: float):
        # PyBullet ignores URDF mimic tags, so drive the six coupled joints explicitly.
        values = (opening, -opening, opening, opening, -opening, opening)
        for name, q in zip(self.finger_names, values):
            p.setJointMotorControl2(self.ur5, self.joints[name], p.POSITION_CONTROL,
                                    targetPosition=q, force=80, maxVelocity=1.5)

    def open_gripper(self):
        self.set_gripper(0.0)

    def close_gripper(self):
        self.set_gripper(0.62)

    def command_pose(self, position, orientation):
        q = p.calculateInverseKinematics(
            self.ur5, self.ee, position, orientation,
            maxNumIterations=100, residualThreshold=1e-5,
        )
        # IK also returns values for movable gripper joints. Use only the six arm
        # values and explicitly enforce the limits from the URDF.
        q = [np.clip(value, p.getJointInfo(self.ur5, j)[8],
                     p.getJointInfo(self.ur5, j)[9])
             for j, value in zip(self.arm_joints, q[:6])]
        q = np.asarray(q)
        if self.robot_a_motion == "baseline":
            self.ur5_command_velocity = (q - self.ur5_command) / DT
            self.ur5_command = q.copy()
        else:
            desired_velocity = np.clip((q - self.ur5_command) / DT,
                                       -ARM_SPEED, ARM_SPEED)
            velocity_change = np.clip(
                desired_velocity - self.ur5_command_velocity,
                -ARM_ACCELERATION * DT, ARM_ACCELERATION * DT,
            )
            self.ur5_command_velocity += velocity_change
            step = self.ur5_command_velocity * DT
            # Do not integrate beyond the current IK target.
            step = np.where(np.abs(step) > np.abs(q - self.ur5_command),
                            q - self.ur5_command, step)
            self.ur5_command += step
        for j, target in zip(self.arm_joints, self.ur5_command):
            info = p.getJointInfo(self.ur5, j)
            p.setJointMotorControl2(
                self.ur5, j, p.POSITION_CONTROL, targetPosition=target,
                force=info[10], maxVelocity=min(ARM_SPEED, info[11]),
                positionGain=0.25,
            )

    def attach_cube(self):
        """Attach at the current measured relative pose (deterministic grasp)."""
        ee_pos, ee_orn = p.getLinkState(self.ur5, self.ee)[4:6]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube)
        inv_pos, inv_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, cube_pos, cube_orn)
        self.cube_in_ee = np.asarray(rel_pos)
        grasp = p.createConstraint(
            self.ur5, self.ee, self.cube, -1, p.JOINT_FIXED, (0, 0, 0),
            rel_pos, (0, 0, 0), rel_orn, (0, 0, 0, 1),
        )
        p.changeConstraint(grasp, maxForce=5000, erp=1.0)
        return grasp

    def ee_goal_for_cube(self, cube_goal, orientation):
        """Convert a desired cube-center position to the required gripper position."""
        offset_world, _ = p.multiplyTransforms(
            (0, 0, 0), orientation, self.cube_in_ee, (0, 0, 0, 1)
        )
        return np.asarray(cube_goal) - np.asarray(offset_world)

    def command_cube_position(self, cube_goal, orientation):
        """Closed-loop Robot-A command that removes IK/motor tracking lag."""
        cube_goal = np.asarray(cube_goal)
        cube_actual = np.asarray(p.getBasePositionAndOrientation(self.cube)[0])
        correction = np.clip(cube_goal - cube_actual, -0.08, 0.08)
        ee_goal = self.ee_goal_for_cube(cube_goal, orientation) + 4.0 * correction
        ee_goal[2] += PAYLOAD_Z_COMPENSATION
        self.command_pose(ee_goal, orientation)

    def settle_cube(self, cube_goal, orientation, duration=0.8):
        for _ in range(round(duration / DT)):
            self.command_cube_position(cube_goal, orientation)
            self.step(DT)

    def move_linear(self, goal, orientation, duration: float):
        start = np.array(p.getLinkState(self.ur5, self.ee)[4])
        goal = np.asarray(goal, dtype=float)
        for k in range(max(1, round(duration / DT))):
            u = smoothstep((k + 1) / round(duration / DT))
            self.command_pose(start + u * (goal - start), orientation)
            self.step(DT)

    def pick_and_transfer(self):
        # The supplied URDF's home pose already points the Robotiq gripper down.
        # Holding this measured orientation avoids assumptions about mesh axes.
        self.stage = "initialization"
        self.step(0.5)
        down = p.getLinkState(self.ur5, self.ee)[5]
        cube_pos = np.array(p.getBasePositionAndOrientation(self.cube)[0])
        self.stage = "approach"
        self.move_linear(cube_pos + (0, 0, 0.22), down, 1.33)
        self.stage = "descend"
        self.move_linear(cube_pos + (0, 0, 0.105), down, 0.80)
        self.stage = "grasp"
        self.close_gripper()
        self.step(0.12)
        # A fixed grasp keeps this assessment demo deterministic; finger motion remains visible.
        self.grasp = self.attach_cube()
        self.step(0.58)
        self.stage = "lift"
        self.move_linear(cube_pos + (0, 0, 0.35), down, 0.93)
        self.stage = "transfer"
        self.move_linear(self.ee_goal_for_cube(PATTERN_CENTER, down), down, 1.67)
        return down

    def execute_topp_joint_schedule(self, name):
        """Execute one offline joint schedule at the physics rate."""
        path = ROOT / "metrics" / "time_optimal" / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python time_optimal.py` first."
            )
        schedule = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        schedule_time = schedule["time"]
        duration = float(schedule_time[-1])
        for index in range(max(1, round(duration / DT)) + 1):
            current_time = min(index * DT, duration)
            joint_target = np.array([
                np.interp(current_time, schedule_time, schedule[f"q{joint + 1}"])
                for joint in range(6)
            ])
            previous_command = self.ur5_command.copy()
            self.ur5_command = joint_target
            self.ur5_command_velocity = (joint_target - previous_command) / DT
            for joint, target in zip(self.arm_joints, joint_target):
                info = p.getJointInfo(self.ur5, joint)
                p.setJointMotorControl2(
                    self.ur5, joint, p.POSITION_CONTROL,
                    targetPosition=float(target), force=info[10],
                    maxVelocity=min(ARM_SPEED, info[11]), positionGain=0.25,
                )
            self.step(DT)
        print(f"Executed TOPP {name}: {duration:.3f} s")

    def pick_and_transfer_topp(self):
        """Execute acceleration-constrained pickup and transfer schedules."""
        self.stage = "initialization"
        self.step(0.5)
        orientation = p.getLinkState(self.ur5, self.ee)[5]
        self.stage = "approach"
        self.execute_topp_joint_schedule("pickup_approach")
        self.stage = "descend"
        self.execute_topp_joint_schedule("pickup_descent")
        self.stage = "grasp"
        self.close_gripper()
        self.step(0.12)
        self.grasp = self.attach_cube()
        self.step(0.58)
        self.stage = "lift"
        self.execute_topp_joint_schedule("pickup_lift")
        self.stage = "transfer"
        self.execute_topp_joint_schedule("pickup_transfer")
        return orientation

    def trace_patterns(self, orientation, cycles: int = 1, speed_scale: float = 1.0):
        self.sync_enabled = True
        center = PATTERN_CENTER
        radius, depth, period = 0.12, 0.04, 8.0 / speed_scale
        first_pattern = True
        for name in ("circle", "lissajous") * cycles:
            print(f"Tracing {name} trajectory")
            steps = round(period / DT)
            if name == "circle":
                first_cube_goal = center + (0, radius, 0)
            else:
                first_cube_goal = center + (0, radius, 0)
            if first_pattern:
                self.stage = f"{name}_setup"
                self.move_linear(
                    self.ee_goal_for_cube(first_cube_goal, orientation), orientation, 1.0
                )
                self.settle_cube(first_cube_goal, orientation)
                first_pattern = False
                self.metrics.request_rotation_reference()
            self.stage = name
            previous_goal = first_cube_goal
            previous_camera = p.getLinkState(self.panda, self.camera_link)[4]
            for k in range(steps + 1):
                # Smooth phase speed avoids a velocity jump at start and finish.
                u = k / steps
                phase = 2 * math.pi * smoothstep(u)
                x = center[0] + depth * math.sin(phase)
                if name == "circle":
                    y = center[1] + radius * math.cos(phase)
                    z = center[2] + radius * math.sin(phase)
                else:
                    # Re-phased figure eight: it shares the circle endpoint.
                    y = center[1] + radius * math.cos(phase)
                    z = center[2] + 0.75 * radius * math.sin(2 * phase)
                cube_goal = (x, y, z)
                # Neutral reference path shown in the GUI.
                color = (0.55, 0.55, 0.55)
                if self.gui and k:
                    p.addUserDebugLine(previous_goal, cube_goal, color, 2.0, 0)
                    current_camera = p.getLinkState(self.panda, self.camera_link)[4]
                    p.addUserDebugLine(
                        previous_camera, current_camera, (0.8, 0.1, 0.8), 2.5, 0
                    )
                    previous_camera = current_camera
                self.command_cube_position(cube_goal, orientation)
                self.step(DT)
                actual = p.getBasePositionAndOrientation(self.cube)[0]
                camera = p.getLinkState(self.panda, self.camera_link)[4]
                self.trajectory_log.append(
                    (name, k * DT, *cube_goal, *actual, *camera)
                )
                previous_goal = cube_goal

    def trace_topp_patterns(self, orientation, cycles: int = 1):
        """Execute offline TOPP joint schedules while Franka tracks visually."""
        self.sync_enabled = True
        self.metrics.request_rotation_reference()
        directory = ROOT / "metrics" / "time_optimal"
        for pattern in ("circle", "lissajous") * cycles:
            path = directory / f"{pattern}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run `python time_optimal.py` first."
                )
            schedule = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
            schedule_time = schedule["time"]
            first_cube_goal = np.array((schedule["x"][0], schedule["y"][0],
                                        schedule["z"][0]))
            self.stage = pattern
            previous_goal = first_cube_goal
            previous_camera = p.getLinkState(self.panda, self.camera_link)[4]
            duration = float(schedule_time[-1])
            steps = max(1, round(duration / DT))
            print(f"Executing TOPP {pattern}: {duration:.3f} s")
            for index in range(steps + 1):
                current_time = min(index * DT, duration)
                joint_target = np.array([
                    np.interp(current_time, schedule_time, schedule[f"q{joint + 1}"])
                    for joint in range(6)
                ])
                previous_command = self.ur5_command.copy()
                self.ur5_command = joint_target
                self.ur5_command_velocity = (joint_target - previous_command) / DT
                for joint, target in zip(self.arm_joints, joint_target):
                    info = p.getJointInfo(self.ur5, joint)
                    p.setJointMotorControl2(
                        self.ur5, joint, p.POSITION_CONTROL,
                        targetPosition=float(target), force=info[10],
                        maxVelocity=min(ARM_SPEED, info[11]), positionGain=0.25,
                    )
                cube_goal = np.array([
                    np.interp(current_time, schedule_time, schedule[axis])
                    for axis in ("x", "y", "z")
                ])
                if self.gui and index:
                    p.addUserDebugLine(previous_goal, cube_goal,
                                       (0.55, 0.55, 0.55), 2.0, 0)
                    current_camera = p.getLinkState(self.panda, self.camera_link)[4]
                    p.addUserDebugLine(previous_camera, current_camera,
                                       (0.8, 0.1, 0.8), 2.5, 0)
                    previous_camera = current_camera
                self.step(DT)
                actual = p.getBasePositionAndOrientation(self.cube)[0]
                camera = p.getLinkState(self.panda, self.camera_link)[4]
                self.trajectory_log.append(
                    (pattern, current_time, *cube_goal, *actual, *camera)
                )
                previous_goal = cube_goal

    def save_trajectory(self, path=None):
        path = path or self.run_dir / f"trajectory_{self.artifact_suffix}.csv"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("pattern", "time", "x_des", "y_des", "z_des",
                             "x", "y", "z", "camera_x", "camera_y", "camera_z"))
            writer.writerows(self.trajectory_log)
        return path

    def run(self, cycles: int, trajectory_speed: float = 1.0):
        print("Picking and transferring cube")
        if self.robot_a_motion == "topp":
            print("Robot A motion: joint-constrained TOPP")
            orientation = self.pick_and_transfer_topp()
            self.trace_topp_patterns(orientation, cycles)
        else:
            orientation = self.pick_and_transfer()
            print(f"Robot A trajectory speed: {trajectory_speed:.2f}x")
            self.trace_patterns(orientation, cycles, trajectory_speed)
        trajectory_path = self.save_trajectory()
        camera_path = self.save_camera_view()
        metrics_path = self.metrics.save(self.run_dir)
        print(f"Saved trajectory to {trajectory_path}")
        if camera_path is not None:
            print(f"Saved Franka camera image to {camera_path}")
        print(f"Saved metrics to {metrics_path}")
        print("Demo complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", action="store_true", help="run without the GUI")
    parser.add_argument("--fast", action="store_true", help="do not pace simulation in wall time")
    parser.add_argument(
        "--video", action="store_true",
        help="record a 30 FPS scene video to simulation_<method>.mp4",
    )
    parser.add_argument(
        "--cycles", type=int, default=2,
        help="number of alternating circle/Lissajous pairs (default: 2)",
    )
    parser.add_argument("--trajectory-speed", type=float, default=2.25,
                        help="Robot A pattern speed multiplier (default: 2.25)")
    parser.add_argument("--robot-a-motion", choices=("baseline", "topp"),
                        default="baseline", help="Robot A timing law")
    parser.add_argument("--method", "--methods", dest="methods", nargs="+",
                        choices=("ik", "kf_vel", "kf_acc", "kf_pose",
                                 "nn_motion", "nn_online", "rls"),
                        default=["ik"], help="one or more synchronization methods")
    args = parser.parse_args()
    completed_methods = []
    for method in dict.fromkeys(args.methods):
        print(f"\n=== Running method: {method} ===")
        demo = Demo(gui=not args.direct, realtime=not args.fast, method=method,
                    robot_a_motion=args.robot_a_motion)
        video_path = demo.run_dir / f"simulation_{demo.artifact_suffix}.mp4"
        if args.video:
            demo.start_video(video_path)
        try:
            demo.run(args.cycles, args.trajectory_speed)
            completed_methods.append(method)
            if not args.direct and not args.fast:
                demo.step(2.0)
        finally:
            if demo.video_writer is not None:
                saved_video = demo.stop_video()
                print(f"Saved simulation video to {saved_video}")
            if p.isConnected():
                p.disconnect()

    if completed_methods:
        from plot import generate as plot_trajectory
        from plot_comparison import generate as plot_comparison
        from plot_metrics import generate as plot_tracking

        for run_dir in sorted((ROOT / "metrics" / "runs").glob("run_*")):
            trajectories = list(run_dir.glob("trajectory_*.csv"))
            samples = list(run_dir.glob("samples_*.csv"))
            if trajectories and samples:
                plot_trajectory(run_dir, show=False)
                plot_tracking(run_dir, show=False)
        comparison_methods = ("ik", "nn_motion", "kf_vel", "kf_acc",
                              "nn_online", "rls")
        if all(all((ROOT / "metrics" / "runs" / f"run_{method}" / filename).exists()
                   for filename in (f"samples_{method}.csv",
                                    f"trajectory_{method}.csv",
                                    f"energy_samples_{method}.csv"))
               for method in comparison_methods):
            plot_comparison(ROOT / "metrics" / "runs")
        print("Regenerated all available method plots using shared scales")


if __name__ == "__main__":
    main()
