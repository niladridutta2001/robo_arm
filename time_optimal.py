"""Offline TOPP planner for Robot A pickup, transfer, and object paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pybullet as p


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 19.2,
    "axes.labelsize": 16,
    "xtick.labelsize": 12.8,
    "ytick.labelsize": 12.8,
    "legend.fontsize": 16,
    "figure.titlesize": 19.2,
})

from main import ARM_ACCELERATION, ARM_SPEED, DT, PATTERN_CENTER, Demo


RADIUS, DEPTH = 0.12, 0.04


def cartesian_path(pattern, phase):
    points = np.empty((len(phase), 3))
    points[:, 0] = PATTERN_CENTER[0] + DEPTH * np.sin(phase)
    if pattern == "circle":
        points[:, 1] = PATTERN_CENTER[1] + RADIUS * np.cos(phase)
        points[:, 2] = PATTERN_CENTER[2] + RADIUS * np.sin(phase)
    else:
        points[:, 1] = PATTERN_CENTER[1] + RADIUS * np.cos(phase)
        points[:, 2] = PATTERN_CENTER[2] + 0.75 * RADIUS * np.sin(2 * phase)
    return points


def arc_length_samples(pattern, count, dense_count=5000):
    dense_phase = np.linspace(0, 2 * np.pi, dense_count + 1)
    dense_points = cartesian_path(pattern, dense_phase)
    length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense_points, axis=0), axis=1))]
    normalized = length / length[-1]
    sample_s = np.linspace(0, 1, count + 1)
    sample_phase = np.interp(sample_s, normalized, dense_phase)
    return sample_s, sample_phase, cartesian_path(pattern, sample_phase), length[-1]


def continuous_ik(demo, points, orientation):
    solutions = []
    previous = np.array([p.getJointState(demo.ur5, j)[0] for j in demo.arm_joints])
    for cube_point in points:
        for joint, value in zip(demo.arm_joints, previous):
            p.resetJointState(demo.ur5, joint, float(value))
        ee_point = demo.ee_goal_for_cube(cube_point, orientation)
        result = p.calculateInverseKinematics(
            demo.ur5, demo.ee, ee_point, orientation,
            maxNumIterations=150, residualThreshold=1e-6,
        )
        previous = np.asarray(result[:6])
        previous = np.array([
            np.clip(value, p.getJointInfo(demo.ur5, joint)[8],
                    p.getJointInfo(demo.ur5, joint)[9])
            for joint, value in zip(demo.arm_joints, previous)
        ])
        solutions.append(previous.copy())
    # Remove artificial +/-pi discontinuities before path differentiation.
    return np.unwrap(np.asarray(solutions), axis=0)


def continuous_ik_ee(demo, points, orientation):
    """Continuous IK for paths specified directly at the end effector."""
    solutions = []
    previous = np.array([p.getJointState(demo.ur5, j)[0]
                         for j in demo.arm_joints])
    for ee_point in points:
        for joint, value in zip(demo.arm_joints, previous):
            p.resetJointState(demo.ur5, joint, float(value))
        result = p.calculateInverseKinematics(
            demo.ur5, demo.ee, ee_point, orientation,
            maxNumIterations=150, residualThreshold=1e-6,
        )
        previous = np.array([
            np.clip(value, p.getJointInfo(demo.ur5, joint)[8],
                    p.getJointInfo(demo.ur5, joint)[9])
            for joint, value in zip(demo.arm_joints, result[:6])
        ])
        solutions.append(previous.copy())
    return np.unwrap(np.asarray(solutions), axis=0)


def acceleration_bounds(q_s, q_ss, path_speed_squared, acceleration_limits):
    lower, upper = -np.inf, np.inf
    for first, second, limit in zip(q_s, q_ss, acceleration_limits):
        offset = second * path_speed_squared
        if abs(first) < 1e-9:
            if abs(offset) > limit:
                return np.inf, -np.inf
            continue
        first_bound = (-limit - offset) / first
        second_bound = (limit - offset) / first
        lower = max(lower, min(first_bound, second_bound))
        upper = min(upper, max(first_bound, second_bound))
    return lower, upper


def topp(q_s, q_ss, ds, velocity_limits, acceleration_limits):
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity_bounds = np.where(np.abs(q_s) > 1e-9,
                                   velocity_limits / np.abs(q_s), np.inf)
    speed_limit = np.min(velocity_bounds, axis=1)
    x_limit = speed_limit ** 2
    x = np.zeros(len(q_s))  # x = s_dot^2

    # Forward pass: maximum reachable speed under positive acceleration.
    for index in range(len(x) - 1):
        _, maximum_acceleration = acceleration_bounds(
            q_s[index], q_ss[index], x[index], acceleration_limits
        )
        if not np.isfinite(maximum_acceleration):
            maximum_acceleration = 0.0
        x[index + 1] = min(x_limit[index + 1],
                           max(0.0, x[index] + 2 * ds * maximum_acceleration))

    # Backward pass: largest speed that can decelerate into the next sample.
    x[-1] = 0.0
    for index in range(len(x) - 2, -1, -1):
        upper_candidate = min(x[index], x_limit[index])
        candidates = np.linspace(0.0, upper_candidate, 500)
        feasible = 0.0
        for candidate in candidates:
            required_acceleration = (x[index + 1] - candidate) / (2 * ds)
            minimum, maximum = acceleration_bounds(
                q_s[index], q_ss[index], candidate, acceleration_limits
            )
            if minimum - 1e-9 <= required_acceleration <= maximum + 1e-9:
                feasible = candidate
        x[index] = feasible

    speed = np.sqrt(np.maximum(x, 0.0))
    segment_time = np.zeros(len(speed) - 1)
    for index, (first, second) in enumerate(zip(speed[:-1], speed[1:])):
        denominator = first + second
        if denominator < 1e-9:
            raise RuntimeError("TOPP produced a zero-speed interior segment")
        segment_time[index] = 2 * ds / denominator
    time = np.r_[0.0, np.cumsum(segment_time)]
    return time, speed


def effort(q_s, speed, time):
    joint_velocity = q_s * speed[:, None]
    return float(np.trapezoid(np.sum(joint_velocity ** 2, axis=1), time)), joint_velocity


def baseline_effort(q_s, duration, samples=2000):
    time = np.linspace(0, duration, samples)
    tau = time / duration
    s = tau * tau * (3 - 2 * tau)
    s_dot = 6 * tau * (1 - tau) / duration
    derivative = np.column_stack([
        np.interp(s, np.linspace(0, 1, len(q_s)), q_s[:, joint])
        for joint in range(q_s.shape[1])
    ])
    joint_velocity = derivative * s_dot[:, None]
    return float(np.trapezoid(np.sum(joint_velocity ** 2, axis=1), time))


def save_schedule(demo, name, points, joints, baseline_duration, acceleration_limit,
                  output):
    """Time-parameterize and save one normalized joint path."""
    count = len(points) - 1
    s = np.linspace(0.0, 1.0, count + 1)
    ds = 1.0 / count
    q_s = np.gradient(joints, ds, axis=0, edge_order=2)
    q_ss = np.gradient(q_s, ds, axis=0, edge_order=2)
    velocity_limits = np.array([
        min(ARM_SPEED, p.getJointInfo(demo.ur5, joint)[11])
        for joint in demo.arm_joints
    ])
    acceleration_limits = np.full(6, acceleration_limit)
    time, speed = topp(q_s, q_ss, ds, velocity_limits, acceleration_limits)
    optimized_effort, joint_velocity = effort(q_s, speed, time)
    original_effort = baseline_effort(q_s, baseline_duration)
    joint_acceleration = np.gradient(joint_velocity, time, axis=0, edge_order=2)
    csv_data = np.column_stack((time, s, points, joints, joint_velocity,
                                joint_acceleration, speed))
    header = ["time", "s", "x", "y", "z"]
    header += [f"q{j + 1}" for j in range(6)]
    header += [f"qd{j + 1}" for j in range(6)]
    header += [f"qdd{j + 1}" for j in range(6)] + ["s_dot"]
    np.savetxt(output / f"{name}.csv", csv_data, delimiter=",",
               header=",".join(header), comments="")
    return {
        "samples": count + 1,
        "baseline_duration_s": baseline_duration,
        "topp_duration_s": float(time[-1]),
        "time_reduction_percent": 100 * (baseline_duration - time[-1]) /
                                  baseline_duration,
        "baseline_velocity_squared_effort": original_effort,
        "topp_velocity_squared_effort": optimized_effort,
        "velocity_limit_rad_s": velocity_limits.tolist(),
        "acceleration_limit_rad_s2": acceleration_limits.tolist(),
    }


def plan_linear_segment(demo, name, start, goal, orientation, count,
                        baseline_duration, acceleration_limit, output):
    interpolation = np.linspace(0.0, 1.0, count + 1)[:, None]
    points = np.asarray(start) + interpolation * (np.asarray(goal) - np.asarray(start))
    joints = continuous_ik_ee(demo, points, orientation)
    return save_schedule(demo, name, points, joints, baseline_duration,
                         acceleration_limit, output)


def plan_pattern(demo, pattern, orientation, count, acceleration_limit, output):
    s, phase, points, path_length = arc_length_samples(pattern, count)
    joints = continuous_ik(demo, points, orientation)
    ds = 1.0 / count
    q_s = np.gradient(joints, ds, axis=0, edge_order=2)
    q_ss = np.gradient(q_s, ds, axis=0, edge_order=2)
    velocity_limits = np.array([
        min(ARM_SPEED, p.getJointInfo(demo.ur5, joint)[11])
        for joint in demo.arm_joints
    ])
    acceleration_limits = np.full(6, acceleration_limit)
    time, speed = topp(q_s, q_ss, ds, velocity_limits, acceleration_limits)
    optimized_effort, joint_velocity = effort(q_s, speed, time)
    baseline_duration = 8.0 / 2.25
    original_effort = baseline_effort(q_s, baseline_duration)

    joint_acceleration = np.gradient(joint_velocity, time, axis=0, edge_order=2)
    csv_data = np.column_stack((time, s, phase, points, joints,
                                joint_velocity, joint_acceleration, speed))
    header = ["time", "s", "phase", "x", "y", "z"]
    header += [f"q{j + 1}" for j in range(6)]
    header += [f"qd{j + 1}" for j in range(6)]
    header += [f"qdd{j + 1}" for j in range(6)] + ["s_dot"]
    np.savetxt(output / f"{pattern}.csv", csv_data, delimiter=",",
               header=",".join(header), comments="")

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(time, joint_velocity)
    axes[0].axhline(ARM_SPEED, color="black", linestyle="--", linewidth=0.8)
    axes[0].axhline(-ARM_SPEED, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Joint velocity [rad/s]")
    axes[1].plot(time, joint_acceleration)
    axes[1].axhline(acceleration_limit, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(-acceleration_limit, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Joint acceleration [rad/s$^2$]")
    axes[2].plot(time, np.sum(joint_velocity ** 2, axis=1), color="tab:purple")
    axes[2].set(xlabel="Time [s]", ylabel=r"Effort proxy $\sum \dot q_j^2$")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / f"{pattern}.png", dpi=180)
    plt.close(figure)
    return {
        "path_length_m": path_length,
        "samples": count + 1,
        "baseline_duration_s": baseline_duration,
        "topp_duration_s": float(time[-1]),
        "time_reduction_percent": 100 * (baseline_duration - time[-1]) / baseline_duration,
        "baseline_velocity_squared_effort": original_effort,
        "topp_velocity_squared_effort": optimized_effort,
        "velocity_limit_rad_s": velocity_limits.tolist(),
        "acceleration_limit_rad_s2": acceleration_limits.tolist(),
    }


def plot_complete_joint_profiles(output, cycles, acceleration_limit):
    """Plot velocity and acceleration over the complete optimized task."""
    sequence = [
        ("initialization", None, 0.5),
        ("approach", "pickup_approach.csv", 0.0),
        ("descent", "pickup_descent.csv", 0.0),
        ("grasp", None, 0.7),
        ("lift", "pickup_lift.csv", 0.0),
        ("transfer", "pickup_transfer.csv", 0.0),
    ]
    sequence.extend((name, f"{name}.csv", 0.0)
                    for name in ("circle", "lissajous") * cycles)

    times, velocities, accelerations, stages = [], [], [], []
    offset = 0.0
    for label, filename, dwell in sequence:
        start = offset
        if filename is None:
            local_time = np.array((0.0, dwell))
            velocity = np.zeros((2, 6))
            acceleration = np.zeros((2, 6))
        else:
            data = np.atleast_1d(np.genfromtxt(
                output / filename, delimiter=",", names=True
            ))
            local_time = data["time"]
            velocity = np.column_stack([data[f"qd{joint}"] for joint in range(1, 7)])
            acceleration = np.column_stack([
                data[f"qdd{joint}"] for joint in range(1, 7)
            ])
        global_time = local_time + offset
        if times:
            global_time = global_time[1:]
            velocity = velocity[1:]
            acceleration = acceleration[1:]
        times.append(global_time)
        velocities.append(velocity)
        accelerations.append(acceleration)
        offset += float(local_time[-1])
        stages.append((label, start, offset))

    time = np.concatenate(times)
    joint_velocity = np.vstack(velocities)
    joint_acceleration = np.vstack(accelerations)
    colors = plt.get_cmap("tab10").colors[:6]
    stage_colors = plt.get_cmap("Pastel1")
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    for joint, color in enumerate(colors):
        axes[0].plot(time, joint_velocity[:, joint], color=color, linewidth=1.2)
        axes[1].plot(time, joint_acceleration[:, joint], color=color, linewidth=1.2)
    axes[0].axhline(ARM_SPEED, color="black", linestyle="--", linewidth=1.2)
    axes[0].axhline(-ARM_SPEED, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(acceleration_limit, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(-acceleration_limit, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Joint velocity [rad/s]")
    axes[1].set(xlabel="Time [s]", ylabel="Joint acceleration [rad/s$^2$]")
    for axis in axes:
        for stage_index, (label, start, stop) in enumerate(stages):
            axis.axvspan(start, stop, color=stage_colors(stage_index % 9),
                         alpha=0.16, linewidth=0)
        axis.grid(True, alpha=0.3)
    for stage_index, (label, start, stop) in enumerate(stages):
        if stop - start > 0.18:
            axes[1].text((start + stop) / 2, 0.02, label,
                         transform=axes[1].get_xaxis_transform(),
                         ha="center", va="bottom", rotation=90, fontsize=10)
    handles = [Line2D([], [], color=colors[joint], label=f"Joint {joint + 1}")
               for joint in range(6)]
    handles.append(Line2D([], [], color="black", linestyle="--",
                          label="Imposed limit"))
    figure.legend(handles=handles, loc="lower center", ncol=7,
                  bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output / "topp_joint_profiles.png", dpi=180)
    plt.close(figure)


def plot_duration_comparison(summary, cycles, output):
    """Plot cumulative baseline and TOPP time at every task boundary."""
    pickup = summary["pickup_and_transfer"]
    labels = ["Start", "Initialization", "Approach", "Descent", "Grasp",
              "Lift", "Transfer", "Pattern setup"]
    baseline_segments = [0.5,
                         pickup["approach"]["baseline_duration_s"],
                         pickup["descent"]["baseline_duration_s"],
                         0.7,
                         pickup["lift"]["baseline_duration_s"],
                         pickup["transfer"]["baseline_duration_s"],
                         1.8]
    topp_segments = [0.5,
                     pickup["approach"]["topp_duration_s"],
                     pickup["descent"]["topp_duration_s"],
                     0.7,
                     pickup["lift"]["topp_duration_s"],
                     pickup["transfer"]["topp_duration_s"],
                     0.0]
    for cycle in range(1, cycles + 1):
        labels.extend((f"Circle {cycle}", f"Lissajous {cycle}"))
        baseline_segments.extend((summary["circle"]["baseline_duration_s"],
                                  summary["lissajous"]["baseline_duration_s"]))
        topp_segments.extend((summary["circle"]["topp_duration_s"],
                              summary["lissajous"]["topp_duration_s"]))

    baseline = np.r_[0.0, np.cumsum(baseline_segments)]
    optimized = np.r_[0.0, np.cumsum(topp_segments)]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(15, 6.5))
    axis.plot(positions, baseline, color="tab:blue", marker="o",
              linewidth=2.0, label="Baseline")
    axis.plot(positions, optimized, color="tab:orange", marker="s",
              linewidth=2.0, label="TOPP")
    axis.fill_between(positions, optimized, baseline, color="tab:green",
                      alpha=0.10)
    axis.annotate(f"{baseline[-1]:.2f} s", (positions[-1], baseline[-1]),
                  xytext=(-8, 8), textcoords="offset points", ha="right")
    axis.annotate(f"{optimized[-1]:.2f} s", (positions[-1], optimized[-1]),
                  xytext=(-8, -18), textcoords="offset points", ha="right")
    axis.set(xlabel="Task boundary", ylabel="Cumulative duration [s]",
             xticks=positions, xticklabels=labels)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(True, axis="y", alpha=0.3)
    figure.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output / "topp_time_comparison.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceleration-limit", type=float, default=ARM_ACCELERATION)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    output = Path("metrics/time_optimal")
    output.mkdir(parents=True, exist_ok=True)
    demo = Demo(gui=False, realtime=False, method="ik")
    try:
        demo.stage = "planning"
        demo.step(0.5)
        orientation = p.getLinkState(demo.ur5, demo.ee)[5]
        cube_position = np.asarray(p.getBasePositionAndOrientation(demo.cube)[0])
        home_position = np.asarray(p.getLinkState(demo.ur5, demo.ee)[4])
        approach_position = cube_position + (0, 0, 0.22)
        grasp_position = cube_position + (0, 0, 0.105)
        lift_position = cube_position + (0, 0, 0.35)

        pickup = {
            "approach": plan_linear_segment(
                demo, "pickup_approach", home_position, approach_position,
                orientation, 120, 1.33, args.acceleration_limit, output
            ),
            "descent": plan_linear_segment(
                demo, "pickup_descent", approach_position, grasp_position,
                orientation, 80, 0.80, args.acceleration_limit, output
            ),
        }

        # Nominal deterministic grasp transform used by both planning and execution.
        inverse_position, inverse_orientation = p.invertTransform(
            grasp_position, orientation
        )
        relative_position, _ = p.multiplyTransforms(
            inverse_position, inverse_orientation, cube_position,
            (0, 0, 0, 1)
        )
        demo.cube_in_ee = np.asarray(relative_position)
        pattern_start = PATTERN_CENTER + (0, RADIUS, 0)
        transfer_position = demo.ee_goal_for_cube(pattern_start, orientation)
        pickup["lift"] = plan_linear_segment(
            demo, "pickup_lift", grasp_position, lift_position,
            orientation, 100, 0.93, args.acceleration_limit, output
        )
        pickup["transfer"] = plan_linear_segment(
            demo, "pickup_transfer", lift_position, transfer_position,
            orientation, 180, 1.67, args.acceleration_limit, output
        )
        summary = {
            "pickup_and_transfer": pickup,
            "circle": plan_pattern(demo, "circle", orientation, 250,
                                   args.acceleration_limit, output),
            "lissajous": plan_pattern(demo, "lissajous", orientation, 500,
                                      args.acceleration_limit, output),
        }
        plot_complete_joint_profiles(output, args.cycles, args.acceleration_limit)
        pickup_baseline = sum(item["baseline_duration_s"] for item in pickup.values())
        pickup_topp = sum(item["topp_duration_s"] for item in pickup.values())
        pattern_baseline = args.cycles * (
            summary["circle"]["baseline_duration_s"]
            + summary["lissajous"]["baseline_duration_s"]
        )
        pattern_topp = args.cycles * (
            summary["circle"]["topp_duration_s"]
            + summary["lissajous"]["topp_duration_s"]
        )
        # Both cases retain 0.5 s initialization and 0.7 s grasp dwell.
        # The baseline additionally uses its 1.8 s first-pattern setup.
        baseline_total = 1.2 + pickup_baseline + 1.8 + pattern_baseline
        topp_total = 1.2 + pickup_topp + pattern_topp
        summary["totals"] = {
            "cycles": args.cycles,
            "baseline_pickup_transfer_duration_s": pickup_baseline,
            "topp_pickup_transfer_duration_s": pickup_topp,
            "baseline_complete_task_duration_s": baseline_total,
            "topp_complete_task_duration_s": topp_total,
            "complete_task_time_reduction_percent": (
                100 * (baseline_total - topp_total) / baseline_total
            ),
        }
        plot_duration_comparison(summary, args.cycles, output)
        with (output / "summary_time_optimal.json").open("w") as file:
            json.dump(summary, file, indent=4)
        print(json.dumps(summary, indent=2))
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
