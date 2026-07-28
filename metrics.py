"""Evaluation-only metrics. Ground truth here is never exposed to the controller."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p


def _mean(values):
    return float(np.mean(values)) if values else None


def _last(values):
    return float(values[-1]) if values else None


class Metrics:
    def __init__(self, ur5, ur5_joints, franka, cube, dt, control_method="ik",
                 ur5_acceleration_limit=None, robot_a_motion="baseline"):
        self.ur5, self.ur5_joints = ur5, ur5_joints
        self.franka, self.franka_joints = franka, list(range(7))
        self.cube, self.dt = cube, dt
        self.started = time.perf_counter()
        self.ur5_energy = self.franka_energy = 0.0
        self.rows = []
        self.energy_rows = []
        self.attempts = self.detections = 0
        self.rejections = {"none": 0, "no_detection": 0}
        self.desired_relative_rotation = None
        self.rotation_reference_pending = False
        self.evaluation_active = False
        self.evaluation_attempts = self.evaluation_detections = 0
        self.control_method = control_method
        self.robot_a_motion = robot_a_motion
        self.physics_steps = 0
        self.ur5_acceleration_limit = ur5_acceleration_limit
        self.previous_ur5_velocity = None
        self.max_ur5_acceleration = 0.0
        self.ur5_acceleration_violations = 0

    @staticmethod
    def _energy(body, joints, dt):
        states = p.getJointStates(body, joints)
        return dt * sum(abs(state[1] * state[3]) for state in states)

    def physics_step(self, stage="unlabelled"):
        self.physics_steps += 1
        self.ur5_energy += self._energy(self.ur5, self.ur5_joints, self.dt)
        self.franka_energy += self._energy(self.franka, self.franka_joints, self.dt)
        velocity = np.array([
            state[1] for state in p.getJointStates(self.ur5, self.ur5_joints)
        ])
        if self.previous_ur5_velocity is not None:
            acceleration = (velocity - self.previous_ur5_velocity) / self.dt
            self.max_ur5_acceleration = max(
                self.max_ur5_acceleration, float(np.max(np.abs(acceleration)))
            )
            if (self.ur5_acceleration_limit is not None and
                    np.any(np.abs(acceleration) > self.ur5_acceleration_limit + 0.05)):
                self.ur5_acceleration_violations += 1
        self.previous_ur5_velocity = velocity
        # Store the energy profile at the 30 Hz plotting/logging rate.
        if self.physics_steps % 8 == 0:
            self.energy_rows.append({
                "time": self.physics_steps * self.dt,
                "stage": stage,
                "ur5_cumulative_energy_j": self.ur5_energy,
                "franka_cumulative_energy_j": self.franka_energy,
            })

    @staticmethod
    def _rotation_matrix(quaternion):
        return np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)

    def request_rotation_reference(self):
        """Capture desired camera-to-cube rotation at the next camera sample."""
        self.rotation_reference_pending = True

    def camera_sample(self, controller, camera_position, forward, up, detected,
                      simulation_time):
        self.attempts += 1
        if not detected:
            self.rejections["no_detection"] += 1
        else:
            self.detections += 1
            self.rejections["none"] += 1

        forward = np.asarray(forward) / np.linalg.norm(forward)
        up = np.asarray(up) - forward * np.dot(up, forward)
        up /= np.linalg.norm(up)
        right, down = np.cross(forward, up), -up
        camera_rotation = np.column_stack((right, down, forward))

        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.cube)
        cube_rotation = self._rotation_matrix(cube_orientation)
        relative_world = np.asarray(cube_position) - np.asarray(camera_position)
        relative_camera = camera_rotation.T @ relative_world
        desired = np.array((0, 0, controller.desired_depth))
        gt_translation_error = float(np.linalg.norm(relative_camera - desired))
        # Evaluation only: project the simulator cube centre into the image.
        # This ground-truth quantity is logged but never exposed to the controller.
        if relative_camera[2] > 1e-9:
            gt_u = controller.cx + controller.fx * relative_camera[0] / relative_camera[2]
            gt_v = controller.cy + controller.fy * relative_camera[1] / relative_camera[2]
            gt_pixel_error = float(np.hypot(gt_u - controller.cx,
                                            gt_v - controller.cy))
        else:
            gt_pixel_error = None

        measured_world = None
        if detected:
            measured_camera = np.array((
                controller.last_error[0], controller.last_error[1],
                controller.last_error[2] + controller.desired_depth,
            ))
            measured_world = np.asarray(camera_position) + camera_rotation @ measured_camera

        relative_rotation = camera_rotation.T @ cube_rotation
        if self.rotation_reference_pending:
            self.desired_relative_rotation = relative_rotation.copy()
            self.rotation_reference_pending = False
            self.evaluation_active = True
        if self.desired_relative_rotation is None:
            gt_rotation_error = None
        else:
            delta = self.desired_relative_rotation.T @ relative_rotation
            gt_rotation_error = math.acos(float(np.clip(
                (np.trace(delta) - 1) / 2, -1, 1
            )))
        raw_pose_rotation_error = None
        if detected and controller.last_visual_object_rotation_world is not None:
            # Diagnostic only: this value is never returned to the controller.
            raw_delta = (controller.last_visual_object_rotation_world.T
                         @ cube_rotation)
            raw_pose_rotation_error = math.acos(float(np.clip(
                (np.trace(raw_delta) - 1) / 2, -1, 1
            )))
        if self.evaluation_active:
            self.evaluation_attempts += 1
            self.evaluation_detections += int(detected)

        self.rows.append({
            "time": simulation_time,
            "evaluation_active": int(self.evaluation_active),
            "pbvs_translation_error_m": (float(np.linalg.norm(controller.last_error))
                                         if detected else None),
            "ground_truth_tracking_error_m": gt_translation_error,
            "ground_truth_rotation_error_rad": gt_rotation_error,
            "visual_rotation_error_rad": (controller.last_visual_rotation_error
                                          if detected else None),
            "raw_pnp_rotation_gt_error_rad": raw_pose_rotation_error,
            "pose_confidence": (controller.pose_confidence if detected else None),
            "pbvs_pixel_error_px": controller.last_pixel_error if detected else None,
            "ground_truth_pixel_error_px": gt_pixel_error,
            "camera_distance_error_m": controller.last_distance_error if detected else None,
            "measured_cube_x": measured_world[0] if detected else None,
            "measured_cube_y": measured_world[1] if detected else None,
            "measured_cube_z": measured_world[2] if detected else None,
            "ground_truth_cube_x": cube_position[0],
            "ground_truth_cube_y": cube_position[1],
            "ground_truth_cube_z": cube_position[2],
            "camera_x": camera_position[0],
            "camera_y": camera_position[1],
            "camera_z": camera_position[2],
            "filtered_cube_x": (controller.estimated_position_world[0]
                                if controller.estimated_position_world is not None else None),
            "filtered_cube_y": (controller.estimated_position_world[1]
                                if controller.estimated_position_world is not None else None),
            "filtered_cube_z": (controller.estimated_position_world[2]
                                if controller.estimated_position_world is not None else None),
            "estimated_frequency_hz": (controller.rls_omega / (2 * np.pi)
                                       if controller.rls_omega is not None else None),
            "nn_online_loss": controller.last_online_loss,
            "nn_online_updates": controller.nn_online_updates,
            "nn_online_prediction_error_m": controller.last_online_prediction_error,
            "nn_online_resets": controller.nn_online_resets,
            "ur5_cumulative_energy_j": self.ur5_energy,
            "franka_cumulative_energy_j": self.franka_energy,
        })

    def save(self, directory="metrics/runs/run_ik"):
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        suffix = output.name.removeprefix("run_")
        if self.rows:
            with (output / f"samples_{suffix}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.rows[0])
                writer.writeheader()
                writer.writerows(self.rows)
        if self.energy_rows:
            with (output / f"energy_samples_{suffix}.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.energy_rows[0])
                writer.writeheader()
                writer.writerows(self.energy_rows)

        def values(key, evaluation_only=False):
            return [row[key] for row in self.rows
                    if row[key] is not None and
                    (not evaluation_only or row["evaluation_active"])]

        pbvs_t = values("pbvs_translation_error_m", evaluation_only=True)
        gt_t = values("ground_truth_tracking_error_m", evaluation_only=True)
        gt_r = values("ground_truth_rotation_error_rad", evaluation_only=True)
        visual_r = values("visual_rotation_error_rad", evaluation_only=True)
        raw_pnp_r = values("raw_pnp_rotation_gt_error_rad", evaluation_only=True)
        pose_confidence = values("pose_confidence", evaluation_only=True)
        pixels = values("pbvs_pixel_error_px", evaluation_only=True)
        gt_pixels = values("ground_truth_pixel_error_px", evaluation_only=True)
        distance = values("camera_distance_error_m", evaluation_only=True)
        frequency = values("estimated_frequency_hz")
        online_loss = values("nn_online_loss")
        online_prediction_error = values("nn_online_prediction_error_m")
        simulated_time = self.physics_steps * self.dt
        summary = {
            "control_method": self.control_method,
            "robot_a_motion": self.robot_a_motion,
            "wall_time_seconds": time.perf_counter() - self.started,
            "simulated_task_time_seconds": simulated_time,
            "number_of_samples": len(self.rows),
            "number_of_evaluation_samples": sum(
                row["evaluation_active"] for row in self.rows
            ),
            "ur5_energy_j": self.ur5_energy,
            "franka_energy_j": self.franka_energy,
            "combined_energy_j": self.ur5_energy + self.franka_energy,
            "average_ur5_mechanical_power_w": (self.ur5_energy / simulated_time
                                                if simulated_time > 0 else None),
            "average_franka_mechanical_power_w": (self.franka_energy / simulated_time
                                                   if simulated_time > 0 else None),
            "average_combined_mechanical_power_w": (
                (self.ur5_energy + self.franka_energy) / simulated_time
                if simulated_time > 0 else None
            ),
            "ur5_acceleration_limit_rad_s2": self.ur5_acceleration_limit,
            "max_measured_ur5_acceleration_rad_s2": self.max_ur5_acceleration,
            "ur5_acceleration_violation_samples": self.ur5_acceleration_violations,
            "mean_pbvs_translation_error_m": _mean(pbvs_t),
            "final_pbvs_translation_error_m": _last(pbvs_t),
            "mean_ground_truth_tracking_error_m": _mean(gt_t),
            "final_ground_truth_tracking_error_m": _last(gt_t),
            "mean_ground_truth_rotation_error_rad": _mean(gt_r),
            "final_ground_truth_rotation_error_rad": _last(gt_r),
            "mean_visual_rotation_error_rad": _mean(visual_r),
            "final_visual_rotation_error_rad": _last(visual_r),
            "mean_raw_pnp_rotation_gt_error_rad": _mean(raw_pnp_r),
            "final_raw_pnp_rotation_gt_error_rad": _last(raw_pnp_r),
            "mean_pose_confidence": _mean(pose_confidence),
            "mean_pbvs_pixel_error_px": _mean(pixels),
            "final_pbvs_pixel_error_px": _last(pixels),
            "mean_ground_truth_pixel_error_px": _mean(gt_pixels),
            "final_ground_truth_pixel_error_px": _last(gt_pixels),
            "mean_absolute_camera_distance_error_m": _mean(distance),
            "final_camera_distance_error_m": _last(distance),
            "estimated_frequency_hz": _last(frequency),
            "final_nn_online_loss": _last(online_loss),
            "nn_online_updates": (self.rows[-1]["nn_online_updates"]
                                  if self.rows else 0),
            "mean_nn_online_prediction_error_m": _mean(online_prediction_error),
            "nn_online_resets": (self.rows[-1]["nn_online_resets"]
                                 if self.rows else 0),
            "visual_detection_rate": (self.evaluation_detections /
                                      self.evaluation_attempts
                                      if self.evaluation_attempts else 0.0),
            "raw_visual_detection_rate": self.detections / self.attempts if self.attempts else 0.0,
            "measurement_rejection_counts": self.rejections,
        }
        summary_path = output / f"summary_{suffix}.json"
        with summary_path.open("w") as file:
            json.dump(summary, file, indent=4)
        return summary_path
