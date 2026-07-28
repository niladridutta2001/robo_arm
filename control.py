"""Minimal image-based joint-space controllers for Robot B."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p


class VisualServoController:
    """RGB-D proportional control with optional constant-velocity filtering."""

    def __init__(self, robot: int, camera_link: int, width=320, height=240,
                 fov_degrees=60.0, desired_depth=0.55, rate=30.0,
                 method="ik"):
        self.robot, self.camera_link, self.method = robot, camera_link, method
        self.width, self.height = width, height
        # PyBullet's computeProjectionMatrixFOV uses vertical FOV. With square
        # pixels and a matching image aspect ratio, fx = fy.
        self.fy = 0.5 * height / np.tan(np.deg2rad(fov_degrees) / 2)
        self.fx, self.cx, self.cy = self.fy, width / 2, height / 2
        self.desired_depth, self.dt = desired_depth, 1.0 / rate
        self.arm = list(range(7))
        self.lower = np.array([p.getJointInfo(robot, j)[8] for j in self.arm])
        self.upper = np.array([p.getJointInfo(robot, j)[9] for j in self.arm])
        self.ranges = self.upper - self.lower
        self.middle = 0.5 * (self.lower + self.upper)
        self.last_error = self.last_pixel_error = self.last_distance_error = None
        self.kf_state = self.kf_covariance = None
        self.pose_translation_state = self.pose_translation_covariance = None
        self.pose_relative_rotation = None
        self.pose_angular_velocity = np.zeros(3)
        self.pose_angular_acceleration = np.zeros(3)
        self.pose_orientation_covariance = None
        self.pose_desired_relative_rotation = None
        self.pose_missed_frames = 0
        self.pose_valid_streak = 0
        self.pose_confidence = 0.0
        self.pose_safe_hand_orientation = None
        self.last_raw_marker_rotation = None
        self.last_visual_object_rotation_world = None
        self.last_visual_rotation_error = None
        self.estimated_position_world = None
        self.rls_time = 0.0
        self.rls_buffer = []
        self.rls_omega = None
        self.rls_harmonics = 15
        self.rls_forgetting = 0.98
        self.rls_theta = None
        self.rls_covariance = None
        self.rls_bad_residuals = 0
        self.rls_samples_since_fit = 0
        self.nn_history = []
        self.nn_model = None
        self.nn_pending_input = None
        self.nn_replay = []
        self.nn_online_steps = 0
        self.nn_online_updates = 0
        self.nn_online_resets = 0
        self.nn_bad_predictions = 0
        self.last_online_loss = None
        self.last_online_prediction_error = None
        if method in ("nn_motion", "nn_online"):
            import torch
            from motion_network import MotionMLP

            model_path = Path(__file__).resolve().parent / "models" / "motion_mlp.pt"
            if not model_path.exists():
                raise FileNotFoundError("Run `python train_nn.py` before using an NN method")
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.nn_window = int(checkpoint["window"])
            self.nn_model = MotionMLP(self.nn_window)
            self.nn_model.load_state_dict(checkpoint["state_dict"])
            self.nn_model.eval()
            self.nn_normalization = checkpoint
            if method == "nn_online":
                # Preserve the pretrained feature extractor and cautiously adapt
                # only the output layer from future visual measurements.
                for parameter in self.nn_model.parameters():
                    parameter.requires_grad_(False)
                for parameter in self.nn_model.network[-1].parameters():
                    parameter.requires_grad_(True)
                self.nn_optimizer = torch.optim.Adam(
                    self.nn_model.network[-1].parameters(), lr=1e-5
                )
                self.nn_initial_output_state = {
                    name: value.detach().clone()
                    for name, value in self.nn_model.network[-1].state_dict().items()
                }

    def _nn_predict(self, position):
        import torch

        window = self.nn_window
        self.nn_history.append(position.copy())
        self.nn_history = self.nn_history[-window:]
        if len(self.nn_history) < window:
            return position, np.zeros(3), np.zeros(3)
        norm = self.nn_normalization
        value = (np.concatenate(self.nn_history) - norm["x_mean"]) / norm["x_std"]
        with torch.no_grad():
            output = self.nn_model(torch.as_tensor(value, dtype=torch.float32)).numpy()
        output = output * norm["y_std"] + norm["y_mean"]
        return output[:3], output[3:6], output[6:9]

    def _nn_online_predict(self, position):
        """Camera-only one-step learning with conservative last-layer updates."""
        import torch

        norm = self.nn_normalization
        # The preceding window predicts this newly arrived visual measurement.
        if self.nn_pending_input is not None:
            with torch.no_grad():
                previous_output = self.nn_model(
                    torch.as_tensor(self.nn_pending_input, dtype=torch.float32)
                ).numpy()
            previous_position = (
                previous_output[:3] * np.asarray(norm["y_std"][:3])
                + np.asarray(norm["y_mean"][:3])
            )
            prediction_error = float(np.linalg.norm(previous_position - position))
            self.last_online_prediction_error = prediction_error
            self.nn_bad_predictions = (self.nn_bad_predictions + 1
                                       if prediction_error > 0.018
                                       else max(0, self.nn_bad_predictions - 1))

            # Detect a changed motion regime from camera measurements only.
            if self.nn_bad_predictions >= 3:
                self.nn_model.network[-1].load_state_dict(
                    self.nn_initial_output_state
                )
                self.nn_optimizer.state.clear()
                self.nn_replay.clear()
                self.nn_history = [position.copy()]
                self.nn_pending_input = None
                self.nn_bad_predictions = 0
                self.nn_online_resets += 1
                self.last_online_loss = None
                return position, np.zeros(3), np.zeros(3)

            target = ((position - np.asarray(norm["y_mean"][:3])) /
                      np.asarray(norm["y_std"][:3]))
            self.nn_replay.append((self.nn_pending_input.copy(), target.astype(np.float32)))
            self.nn_replay = self.nn_replay[-128:]

        self.nn_history.append(position.copy())
        self.nn_history = self.nn_history[-self.nn_window:]
        if len(self.nn_history) < self.nn_window:
            return position, np.zeros(3), np.zeros(3)

        value = ((np.concatenate(self.nn_history) - np.asarray(norm["x_mean"])) /
                 np.asarray(norm["x_std"])).astype(np.float32)
        self.nn_pending_input = value.copy()
        self.nn_online_steps += 1

        # Adapt infrequently to keep learning outside the critical control path.
        if self.nn_online_steps % 5 == 0 and len(self.nn_replay) >= 16:
            count = min(32, len(self.nn_replay))
            indices = np.linspace(0, len(self.nn_replay) - 1, count, dtype=int)
            batch_x = torch.as_tensor(
                np.stack([self.nn_replay[i][0] for i in indices]),
                dtype=torch.float32,
            )
            batch_y = torch.as_tensor(
                np.stack([self.nn_replay[i][1] for i in indices]),
                dtype=torch.float32,
            )
            self.nn_model.train()
            self.nn_optimizer.zero_grad()
            prediction = self.nn_model(batch_x)[:, :3]
            loss = torch.mean((prediction - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.nn_model.network[-1].parameters(), max_norm=0.5
            )
            self.nn_optimizer.step()
            self.nn_model.eval()
            self.last_online_loss = float(loss.detach())
            self.nn_online_updates += 1

        with torch.no_grad():
            normalized = self.nn_model(torch.as_tensor(value)).numpy()
        output = (normalized * np.asarray(norm["y_std"]) +
                  np.asarray(norm["y_mean"]))
        predicted_next = output[:3]
        velocity = (predicted_next - position) / self.dt
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > 0.50:
            velocity *= 0.50 / velocity_norm
        return predicted_next, velocity, np.zeros(3)

    def _fourier_features(self, current_time):
        features, derivatives = [1.0], [0.0]
        for harmonic in range(1, self.rls_harmonics + 1):
            frequency = harmonic * self.rls_omega
            features.extend((np.sin(frequency * current_time),
                             np.cos(frequency * current_time)))
            derivatives.extend((frequency * np.cos(frequency * current_time),
                                -frequency * np.sin(frequency * current_time)))
        return np.asarray(features), np.asarray(derivatives)

    def _rls_update(self, current_time, position):
        features, _ = self._fourier_features(current_time)
        covariance_features = self.rls_covariance @ features
        gain = covariance_features / (
            self.rls_forgetting + features @ covariance_features
        )
        error = position - features @ self.rls_theta
        self.rls_theta += np.outer(gain, error)
        self.rls_covariance = (
            self.rls_covariance - np.outer(gain, features) @ self.rls_covariance
        ) / self.rls_forgetting

    def _initialize_rls(self, samples):
        feature_matrix = np.vstack([
            self._fourier_features(sample_time)[0]
            for sample_time, _ in samples
        ])
        positions = np.vstack([position for _, position in samples])
        regularized = feature_matrix.T @ feature_matrix + 1e-4 * np.eye(
            feature_matrix.shape[1]
        )
        self.rls_theta = np.linalg.solve(
            regularized, feature_matrix.T @ positions
        )
        self.rls_covariance = np.linalg.inv(regularized)
        self.rls_bad_residuals = 0
        self.rls_samples_since_fit = 0

    def _estimate_rls_frequency(self):
        if len(self.rls_buffer) < 120:
            return False
        times = np.asarray([sample[0] for sample in self.rls_buffer])
        positions = np.asarray([sample[1] for sample in self.rls_buffer])
        axis = int(np.argmax(np.var(positions, axis=0)))
        signal = positions[:, axis] - np.mean(positions[:, axis])
        windowed = signal * np.hanning(len(signal))
        transform_size = 4096
        frequencies = np.fft.rfftfreq(transform_size, d=self.dt)
        spectrum = np.abs(np.fft.rfft(windowed, n=transform_size))
        valid = (frequencies >= 0.10) & (frequencies <= 2.0)
        if not np.any(valid):
            return False
        valid_indices = np.flatnonzero(valid)
        peak = valid_indices[np.argmax(spectrum[valid])]
        # Quadratic spectral interpolation gives sub-bin frequency resolution.
        delta = 0.0
        if 0 < peak < len(spectrum) - 1:
            left, middle, right = np.log(spectrum[peak - 1:peak + 2] + 1e-12)
            denominator = left - 2 * middle + right
            if abs(denominator) > 1e-12:
                delta = 0.5 * (left - right) / denominator
        fft_frequency = (peak + np.clip(delta, -0.5, 0.5)) / (
            transform_size * self.dt
        )
        # Use autocorrelation as a consistency check when it agrees with FFT.
        correlation = np.correlate(signal, signal, mode="full")[len(signal) - 1:]
        minimum_lag = max(1, int(1.0 / (1.0 * self.dt)))
        maximum_lag = min(len(signal) - 1, int(1.0 / (0.15 * self.dt)))
        lag = minimum_lag + int(np.argmax(correlation[minimum_lag:maximum_lag + 1]))
        correlation_frequency = 1.0 / (lag * self.dt)
        frequency = fft_frequency
        if abs(correlation_frequency - fft_frequency) / fft_frequency < 0.20:
            frequency = 0.5 * (fft_frequency + correlation_frequency)
        self.rls_omega = 2.0 * np.pi * frequency
        self._initialize_rls(self.rls_buffer)
        return True

    def _rls_predict(self, current_time, measured_position=None):
        if self.rls_omega is None:
            if measured_position is not None:
                self.rls_buffer.append((current_time, measured_position.copy()))
                self._estimate_rls_frequency()
            if self.rls_omega is None:
                return measured_position, np.zeros(3)
        elif measured_position is not None:
            self.rls_buffer.append((current_time, measured_position.copy()))
            self.rls_buffer = self.rls_buffer[-120:]
            features, _ = self._fourier_features(current_time)
            residual = np.linalg.norm(measured_position - features @ self.rls_theta)
            self.rls_bad_residuals = (self.rls_bad_residuals + 1
                                      if residual > 0.03
                                      else max(0, self.rls_bad_residuals - 1))
            if self.rls_bad_residuals >= 8:
                self._initialize_rls(self.rls_buffer[-30:])
            self._rls_update(current_time, measured_position)
            self.rls_samples_since_fit += 1
        features, derivatives = self._fourier_features(current_time)
        velocity = derivatives @ self.rls_theta
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > 0.50:
            velocity *= 0.50 / velocity_norm
        return features @ self.rls_theta, velocity

    @staticmethod
    def metric_depth(buffer_depth):
        near, far = 0.05, 3.0
        return far * near / (far - (far - near) * buffer_depth)

    def observe(self, rgb, depth):
        self.last_red_roi = None
        red = ((rgb[..., 0] > 180) & (rgb[..., 1] < 105) &
               (rgb[..., 2] < 105) & (rgb[..., 0] > 2.0 * rgb[..., 1]) &
               (rgb[..., 0] > 2.0 * rgb[..., 2]))
        v, u = np.nonzero(red)
        if len(u) < 12:
            return None
        z_values = self.metric_depth(depth[v, u])
        valid = np.isfinite(z_values) & (z_values < 2.5)
        if valid.sum() < 12:
            return None
        u, v, z_values = u[valid], v[valid], z_values[valid]
        z = float(np.median(z_values))
        mean_u, mean_v = float(np.mean(u)), float(np.mean(v))
        margin = 12
        self.last_red_centroid = (mean_u, mean_v)
        self.last_red_roi = (max(0, int(v.min()) - margin),
                             min(self.height, int(v.max()) + margin + 1),
                             max(0, int(u.min()) - margin),
                             min(self.width, int(u.max()) + margin + 1))
        self.last_pixel_error = float(np.hypot(mean_u - self.cx, mean_v - self.cy))
        self.last_distance_error = abs(z - self.desired_depth)
        return np.array(((mean_u - self.cx) * z / self.fx,
                         (mean_v - self.cy) * z / self.fy,
                         z - self.desired_depth))

    def observe_marker_pose(self, rgb, depth):
        """Estimate cube-to-camera orientation with ArUco PnP and confidence."""
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, identifiers, _ = cv2.aruco.ArucoDetector(
                dictionary, parameters
            ).detectMarkers(gray)
        else:
            corners, identifiers, _ = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters
            )
        if identifiers is None or 23 not in identifiers.flatten():
            self.pose_confidence = 0.0
            return None
        index = int(np.flatnonzero(identifiers.flatten() == 23)[0])
        image_points = np.asarray(corners[index], dtype=np.float64).reshape(4, 2)
        area = abs(float(cv2.contourArea(image_points.astype(np.float32))))
        if area < 30.0:
            self.pose_confidence = 0.0
            return None
        half_size, face_x = 0.0176, 0.0282
        object_points = np.array((
            (face_x, -half_size, half_size),
            (face_x, half_size, half_size),
            (face_x, half_size, -half_size),
            (face_x, -half_size, -half_size),
        ), dtype=np.float64)
        camera_matrix = np.array(((self.fx, 0, self.cx),
                                  (0, self.fy, self.cy),
                                  (0, 0, 1)), dtype=np.float64)
        success, rotation_vector, translation = cv2.solvePnP(
            object_points, image_points, camera_matrix, np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or translation[2, 0] <= 0:
            self.pose_confidence = 0.0
            return None
        projected, _ = cv2.projectPoints(
            object_points, rotation_vector, translation, camera_matrix, np.zeros(5)
        )
        residual = projected.reshape(4, 2) - image_points
        reprojection_rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        confidence = min(1.0, area / 120.0) * np.exp(-0.5 * (reprojection_rms / 2.0) ** 2)
        rotation, _ = cv2.Rodrigues(rotation_vector)
        if self.last_raw_marker_rotation is not None:
            jump = np.linalg.norm(self._rotation_vector(
                rotation @ self.last_raw_marker_rotation.T
            ))
            if jump > np.deg2rad(25):
                self.pose_confidence = 0.0
                return None
        if confidence < 0.25:
            self.pose_confidence = float(confidence)
            return None
        self.last_raw_marker_rotation = rotation.copy()
        self.pose_confidence = float(confidence)
        return translation.ravel(), rotation

    @staticmethod
    def _rotation_vector(rotation):
        cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(cosine))
        if angle < 1e-8:
            return 0.5 * np.array((rotation[2, 1] - rotation[1, 2],
                                   rotation[0, 2] - rotation[2, 0],
                                   rotation[1, 0] - rotation[0, 1]))
        axis = np.array((rotation[2, 1] - rotation[1, 2],
                         rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1])) / (2.0 * np.sin(angle))
        return angle * axis

    @staticmethod
    def _rotation_matrix(rotation_vector):
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-10:
            return np.eye(3)
        axis = rotation_vector / angle
        skew = np.array(((0, -axis[2], axis[1]),
                         (axis[2], 0, -axis[0]),
                         (-axis[1], axis[0], 0)))
        return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)

    @staticmethod
    def _quaternion(rotation):
        trace = float(np.trace(rotation))
        if trace > 0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = ((rotation[2, 1] - rotation[1, 2]) / scale,
                          (rotation[0, 2] - rotation[2, 0]) / scale,
                          (rotation[1, 0] - rotation[0, 1]) / scale,
                          0.25 * scale)
        else:
            index = int(np.argmax(np.diag(rotation)))
            if index == 0:
                scale = 2.0 * np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
                quaternion = (0.25 * scale,
                              (rotation[0, 1] + rotation[1, 0]) / scale,
                              (rotation[0, 2] + rotation[2, 0]) / scale,
                              (rotation[2, 1] - rotation[1, 2]) / scale)
            elif index == 1:
                scale = 2.0 * np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
                quaternion = ((rotation[0, 1] + rotation[1, 0]) / scale,
                              0.25 * scale,
                              (rotation[1, 2] + rotation[2, 1]) / scale,
                              (rotation[0, 2] - rotation[2, 0]) / scale)
            else:
                scale = 2.0 * np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
                quaternion = ((rotation[0, 2] + rotation[2, 0]) / scale,
                              (rotation[1, 2] + rotation[2, 1]) / scale,
                              0.25 * scale,
                              (rotation[1, 0] - rotation[0, 1]) / scale)
        quaternion = np.asarray(quaternion)
        return tuple(quaternion / np.linalg.norm(quaternion))

    def _predict(self):
        if self.kf_state is None:
            return
        eye = np.eye(3)
        if self.method == "kf_acc":
            zero = np.zeros((3, 3))
            transition = np.block([
                [eye, self.dt * eye, 0.5 * self.dt ** 2 * eye],
                [zero, eye, self.dt * eye],
                [zero, zero, eye],
            ])
            noise_map = np.vstack(((self.dt ** 3 / 6) * eye,
                                   0.5 * self.dt ** 2 * eye,
                                   self.dt * eye))
            process_noise = 100.0 * noise_map @ noise_map.T
        else:
            transition = np.block([[eye, self.dt * eye],
                                   [np.zeros((3, 3)), eye]])
            noise_map = np.vstack((0.5 * self.dt ** 2 * eye, self.dt * eye))
            process_noise = 9.0 * noise_map @ noise_map.T
        self.kf_state = transition @ self.kf_state
        self.kf_covariance = (transition @ self.kf_covariance @ transition.T
                              + process_noise)

    def _correct(self, position):
        if self.kf_state is None:
            if self.method == "kf_acc":
                self.kf_state = np.r_[position, (0, 0, 0), (0, 0, 0)].astype(float)
                self.kf_covariance = np.diag([0.01] * 3 + [0.25] * 3 + [1.0] * 3)
            else:
                self.kf_state = np.r_[position, (0, 0, 0)].astype(float)
                self.kf_covariance = np.diag([0.01] * 3 + [0.25] * 3)
            return True
        blocks = 3 if self.method == "kf_acc" else 2
        observation = np.hstack((np.eye(3), np.zeros((3, 3 * (blocks - 1)))))
        noise = np.diag((0.006 ** 2, 0.006 ** 2, 0.012 ** 2))
        innovation = position - observation @ self.kf_state
        covariance = observation @ self.kf_covariance @ observation.T + noise
        if float(innovation.T @ np.linalg.solve(covariance, innovation)) > 25.0:
            return False
        gain = self.kf_covariance @ observation.T @ np.linalg.inv(covariance)
        self.kf_state += gain @ innovation
        self.kf_covariance = (np.eye(3 * blocks) - gain @ observation) @ self.kf_covariance
        return True

    def _pose_predict(self):
        """Predict translation and the nominal relative rotation (MEKF)."""
        if self.pose_translation_state is None:
            return
        eye, zero = np.eye(3), np.zeros((3, 3))
        transition = np.block([
            [eye, self.dt * eye, 0.5 * self.dt ** 2 * eye],
            [zero, eye, self.dt * eye],
            [zero, zero, eye],
        ])
        noise_map = np.vstack(((self.dt ** 3 / 6) * eye,
                               0.5 * self.dt ** 2 * eye, self.dt * eye))
        self.pose_translation_state = transition @ self.pose_translation_state
        self.pose_translation_covariance = (
            transition @ self.pose_translation_covariance @ transition.T
            + 100.0 * noise_map @ noise_map.T
        )
        increment = (self.dt * self.pose_angular_velocity
                     + 0.5 * self.dt ** 2 * self.pose_angular_acceleration)
        self.pose_relative_rotation = (
            self._rotation_matrix(increment) @ self.pose_relative_rotation
        )
        self.pose_angular_velocity += self.dt * self.pose_angular_acceleration
        self.pose_orientation_covariance = (
            transition @ self.pose_orientation_covariance @ transition.T
            + 2.0 * noise_map @ noise_map.T
        )

    def _pose_correct(self, position, measured_rotation, confidence):
        """Correct position and rotation; rotation uses a multiplicative error."""
        eye = np.eye(3)
        observation = np.hstack((eye, np.zeros((3, 6))))
        if self.pose_translation_state is None:
            self.pose_translation_state = np.r_[position, np.zeros(6)]
            self.pose_translation_covariance = np.diag(
                [0.01] * 3 + [0.25] * 3 + [1.0] * 3
            )
            self.pose_relative_rotation = measured_rotation.copy()
            self.pose_orientation_covariance = np.diag(
                [np.deg2rad(4) ** 2] * 3
                + [np.deg2rad(25) ** 2] * 3
                + [np.deg2rad(60) ** 2] * 3
            )
            return True

        position_noise = np.diag((0.006 ** 2, 0.006 ** 2, 0.012 ** 2)) / confidence
        position_innovation = position - observation @ self.pose_translation_state
        position_covariance = (observation @ self.pose_translation_covariance
                               @ observation.T + position_noise)
        if float(position_innovation.T @ np.linalg.solve(
                position_covariance, position_innovation)) > 25.0:
            return False

        rotation_innovation = self._rotation_vector(
            measured_rotation @ self.pose_relative_rotation.T
        )
        if np.linalg.norm(rotation_innovation) > np.deg2rad(20):
            return False
        rotation_noise = eye * (np.deg2rad(1.5) ** 2 / confidence)
        rotation_covariance = (observation @ self.pose_orientation_covariance
                               @ observation.T + rotation_noise)
        if float(rotation_innovation.T @ np.linalg.solve(
                rotation_covariance, rotation_innovation)) > 25.0:
            return False

        gain = (self.pose_translation_covariance @ observation.T
                @ np.linalg.inv(position_covariance))
        self.pose_translation_state += gain @ position_innovation
        self.pose_translation_covariance = (
            (np.eye(9) - gain @ observation) @ self.pose_translation_covariance
        )
        gain = (self.pose_orientation_covariance @ observation.T
                @ np.linalg.inv(rotation_covariance))
        correction = gain @ rotation_innovation
        self.pose_relative_rotation = (
            self._rotation_matrix(correction[:3]) @ self.pose_relative_rotation
        )
        self.pose_angular_velocity += correction[3:6]
        self.pose_angular_acceleration += correction[6:9]
        self.pose_orientation_covariance = (
            (np.eye(9) - gain @ observation) @ self.pose_orientation_covariance
        )
        return True

    def update(self, rgb, depth, forward_world, up_world):
        if self.method == "rls":
            self.rls_time += self.dt
        measurement = self.observe(rgb, depth)
        detected = measurement is not None
        if not detected:
            self.last_error = self.last_pixel_error = self.last_distance_error = None
            if self.method == "nn_online":
                self.nn_pending_input = None
            rls_can_predict = self.method == "rls" and self.rls_omega is not None
            if self.method not in ("kf_vel", "kf_acc", "kf_pose") and not rls_can_predict:
                return False

        forward = np.asarray(forward_world, dtype=float)
        forward /= np.linalg.norm(forward)
        up = np.asarray(up_world, dtype=float)
        up -= forward * np.dot(up, forward)
        up /= np.linalg.norm(up)
        right, down = np.cross(forward, up), -up
        right /= np.linalg.norm(right)
        camera_rotation = np.column_stack((right, down, forward))
        camera_position, hand_orientation = p.getLinkState(
            self.robot, self.camera_link
        )[4:6]
        camera_position = np.asarray(camera_position)
        angular_velocity_command = None

        if self.method == "kf_pose":
            marker_pose = self.observe_marker_pose(rgb, depth) if detected else None
            self._pose_predict()
            accepted = False
            if marker_pose is not None:
                _, object_rotation_camera = marker_pose
                # Red RGB-D supplies robust translation; marker correspondences
                # supply orientation. Add the known half-size to move from the
                # visible face to the cube centre.
                relative_position = np.array((measurement[0], measurement[1],
                                              measurement[2]
                                              + self.desired_depth + 0.0275))
                object_position_world = (camera_position
                                         + camera_rotation @ relative_position)
                measurement = np.array((relative_position[0], relative_position[1],
                                        relative_position[2] - self.desired_depth))
                self.last_error = measurement
                self.last_distance_error = abs(measurement[2])
                accepted = self._pose_correct(
                    object_position_world, object_rotation_camera,
                    max(self.pose_confidence, 0.25),
                )
                if accepted:
                    self.last_visual_object_rotation_world = (
                        camera_rotation @ object_rotation_camera
                    )
                    if self.pose_desired_relative_rotation is None:
                        self.pose_desired_relative_rotation = object_rotation_camera.copy()
            detected = accepted
            if accepted:
                self.pose_missed_frames = 0
                self.pose_valid_streak += 1
            else:
                self.pose_missed_frames += 1
                self.pose_valid_streak = 0
                self.pose_confidence = 0.0
            if (self.pose_translation_state is None
                    or self.pose_desired_relative_rotation is None):
                return False
            self.estimated_position_world = self.pose_translation_state[:3].copy()
            predicted_velocity = (self.pose_translation_state[3:6]
                                  + self.dt * self.pose_translation_state[6:9])
            desired_camera = (self.estimated_position_world
                              - forward * self.desired_depth)
            velocity = (2.8 * (desired_camera - camera_position)
                        + 0.8 * predicted_velocity)

            relative_rotation_error = self._rotation_vector(
                self.pose_relative_rotation @ self.pose_desired_relative_rotation.T
            )
            self.last_visual_rotation_error = float(
                np.linalg.norm(relative_rotation_error)
            )
            # Require a stable run of accepted PnP poses before angular control.
            if self.pose_valid_streak >= 12:
                relative_angular_velocity = (
                    self.pose_angular_velocity
                    + self.dt * self.pose_angular_acceleration
                )
                local_angular_velocity = (0.8 * relative_rotation_error
                                          + 0.10 * relative_angular_velocity)
                angular_velocity_command = camera_rotation @ local_angular_velocity
                angular_speed = np.linalg.norm(angular_velocity_command)
                if angular_speed > 0.25:
                    angular_velocity_command *= 0.25 / angular_speed
        elif self.method in ("kf_vel", "kf_acc"):
            self._predict()
            if detected:
                relative = np.array((measurement[0], measurement[1],
                                     measurement[2] + self.desired_depth))
                self._correct(camera_position + camera_rotation @ relative)
            if self.kf_state is None:
                return False
            self.estimated_position_world = self.kf_state[:3].copy()
            predicted_velocity = self.kf_state[3:6]
            if self.method == "kf_acc":
                predicted_velocity = predicted_velocity + self.dt * self.kf_state[6:9]
            desired_camera = (self.estimated_position_world
                              - forward * self.desired_depth)
            velocity = (2.8 * (desired_camera - camera_position)
                        + 0.8 * predicted_velocity)
        elif self.method in ("nn_motion", "nn_online"):
            relative = np.array((measurement[0], measurement[1],
                                 measurement[2] + self.desired_depth))
            measured_world = camera_position + camera_rotation @ relative
            if self.method == "nn_online":
                position, object_velocity, object_acceleration = self._nn_online_predict(
                    measured_world
                )
            else:
                position, object_velocity, object_acceleration = self._nn_predict(
                    measured_world
                )
            self.estimated_position_world = position.copy()
            desired_camera = position - forward * self.desired_depth
            if self.method == "nn_online":
                feedforward_gain = 0.35 * min(1.0, self.nn_online_updates / 20.0)
            else:
                feedforward_gain = 0.8
            velocity = (2.8 * (desired_camera - camera_position)
                        + feedforward_gain *
                        (object_velocity + self.dt * object_acceleration))
        elif self.method == "rls":
            measured_world = None
            if detected:
                relative = np.array((measurement[0], measurement[1],
                                     measurement[2] + self.desired_depth))
                measured_world = camera_position + camera_rotation @ relative
            position, object_velocity = self._rls_predict(
                self.rls_time, measured_world
            )
            if position is None:
                return False
            self.estimated_position_world = position.copy()
            desired_camera = position - forward * self.desired_depth
            feedforward_ramp = min(1.0, self.rls_samples_since_fit / 30.0)
            velocity = (2.8 * (desired_camera - camera_position)
                        + feedforward_ramp * 0.8 * object_velocity)
        else:
            self.estimated_position_world = None
            velocity = 2.8 * camera_rotation @ measurement

        speed = np.linalg.norm(velocity)
        if speed > 0.50:
            velocity *= 0.50 / speed

        q = np.array([p.getJointState(self.robot, j)[0] for j in self.arm])
        if self.method == "kf_pose":
            hand_rotation = np.asarray(
                p.getMatrixFromQuaternion(hand_orientation)
            ).reshape(3, 3)
            if self.pose_safe_hand_orientation is None:
                self.pose_safe_hand_orientation = hand_orientation
            if detected and angular_velocity_command is not None:
                hand_to_camera = hand_rotation.T @ camera_rotation
                target_camera_rotation = (
                    self._rotation_matrix(self.dt * angular_velocity_command)
                    @ camera_rotation
                )
                target_hand_rotation = target_camera_rotation @ hand_to_camera.T
                target_orientation = self._quaternion(target_hand_rotation)
                self.pose_safe_hand_orientation = target_orientation
            else:
                # Any missed/rejected pose immediately freezes orientation.
                target_orientation = self.pose_safe_hand_orientation
        else:
            if not hasattr(self, "desired_hand_orientation"):
                self.desired_hand_orientation = hand_orientation
            target_orientation = self.desired_hand_orientation
        target_position = camera_position + self.dt * velocity
        rest = 0.85 * q + 0.15 * self.middle
        solution = p.calculateInverseKinematics(
            self.robot, self.camera_link, target_position,
            target_orientation,
            lowerLimits=self.lower.tolist(), upperLimits=self.upper.tolist(),
            jointRanges=self.ranges.tolist(), restPoses=rest.tolist(),
            maxNumIterations=80, residualThreshold=1e-5,
        )
        target = np.clip(np.asarray(solution[:7]),
                         self.lower + 0.02, self.upper - 0.02)
        for j, value in zip(self.arm, target):
            info = p.getJointInfo(self.robot, j)
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL, targetPosition=value,
                force=info[10], maxVelocity=min(1.2, info[11]), positionGain=0.25,
            )
        self.last_error = measurement
        return detected
