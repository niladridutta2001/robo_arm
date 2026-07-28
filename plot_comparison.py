"""Create publication-ready 2x3 comparisons for the six retained methods."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


METHODS = ("ik", "nn_motion", "kf_vel", "kf_acc", "nn_online", "rls")
LABELS = {
    "ik": "Proportional IK",
    "kf_vel": "KF velocity",
    "kf_acc": "KF acceleration",
    "kf_pose": "KF pose",
    "nn_motion": "Offline neural",
    "nn_online": "Online neural",
    "rls": "Fourier RLS",
}

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 12.8,
    "ytick.labelsize": 12.8,
    "legend.fontsize": 14,
})


def _load_csv(path, text=False):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; rerun method {path.parent.name}.")
    options = {"dtype": None, "encoding": "utf-8"} if text else {}
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, **options))


def _paths(root, method):
    directory = root / f"run_{method}"
    return directory, method


def _evaluation_samples(root, method):
    directory, suffix = _paths(root, method)
    data = _load_csv(directory / f"samples_{suffix}.csv")
    if "evaluation_active" in data.dtype.names:
        data = data[data["evaluation_active"] > 0.5]
    return data


def _method_label(axis, method):
    options = {
        "transform": axis.transAxes,
        "ha": "left", "va": "top", "fontsize": 14,
        "bbox": {"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    }
    if hasattr(axis, "text2D"):
        axis.text2D(0.02, 0.96, LABELS[method], **options)
    else:
        axis.text(0.02, 0.96, LABELS[method], **options)


def trajectories(root, output):
    loaded = []
    all_points = []
    for method in METHODS:
        directory, suffix = _paths(root, method)
        path = _load_csv(directory / f"trajectory_{suffix}.csv", text=True)
        samples = _evaluation_samples(root, method)
        loaded.append((method, path, samples))
        for names, source in ((('x', 'y', 'z'), path),
                              (('camera_x', 'camera_y', 'camera_z'), path),
                              (('measured_cube_x', 'measured_cube_y',
                                'measured_cube_z'), samples)):
            points = np.column_stack([source[name] for name in names])
            all_points.append(points[np.all(np.isfinite(points), axis=1)])
    cloud = np.vstack(all_points)
    low, high = np.min(cloud, axis=0), np.max(cloud, axis=0)
    padding = np.maximum(0.03, 0.05 * (high - low))

    figure = plt.figure(figsize=(18, 11))
    for index, (method, path, samples) in enumerate(loaded, 1):
        axis = figure.add_subplot(2, 3, index, projection="3d")
        axis.plot(path["x"], path["y"], path["z"], color="tab:blue",
                  linewidth=1.5)
        valid = np.isfinite(samples["measured_cube_x"])
        axis.plot(samples["measured_cube_x"][valid],
                  samples["measured_cube_y"][valid],
                  samples["measured_cube_z"][valid], color="black",
                  linestyle="none", marker=".", markersize=2.5)
        axis.plot(path["camera_x"], path["camera_y"], path["camera_z"],
                  color="tab:purple", linewidth=1.2, alpha=0.8)
        axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
        axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
        axis.set_zlim(low[2] - padding[2], high[2] + padding[2])
        _method_label(axis, method)
    handles = (
        Line2D([], [], color="tab:blue", label="Ground-truth cube"),
        Line2D([], [], color="black", marker=".", linestyle="none",
               label="Camera-measured cube"),
        Line2D([], [], color="tab:purple", label="Franka camera"),
    )
    figure.legend(handles=handles, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output / "trajectories_comparison.png", dpi=180)
    plt.close(figure)


def tracking_errors(root, output):
    datasets = [_evaluation_samples(root, method) for method in METHODS]
    translation_max = max(np.nanmax(np.r_[data["pbvs_translation_error_m"],
                                         data["ground_truth_tracking_error_m"]])
                          for data in datasets) * 1050
    rotation_max = max(np.nanmax(np.degrees(data["ground_truth_rotation_error_rad"]))
                       for data in datasets) * 1.05
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    for axis, method, data in zip(axes.flat, METHODS, datasets):
        time = data["time"] - data["time"][0]
        axis.plot(time, 1000 * data["pbvs_translation_error_m"],
                  color="tab:blue", linewidth=1.3)
        axis.plot(time, 1000 * data["ground_truth_tracking_error_m"],
                  color="tab:orange", linewidth=1.2)
        axis.set_ylim(0, translation_max)
        axis.set_xlabel("Simulation time [s]")
        if axis in axes[:, 0]:
            axis.set_ylabel("Translation error [mm]")
        axis.grid(True, alpha=0.25)
        rotation_axis = axis.twinx()
        rotation_axis.plot(time, np.degrees(data["ground_truth_rotation_error_rad"]),
                           color="tab:red", linewidth=1.1, alpha=0.8)
        rotation_axis.set_ylim(0, rotation_max)
        if axis in axes[:, -1]:
            rotation_axis.set_ylabel("Rotation error [deg]")
        else:
            rotation_axis.set_yticklabels([])
        _method_label(axis, method)
    handles = (
        Line2D([], [], color="tab:blue", label="Visual translation error"),
        Line2D([], [], color="tab:orange", label="Ground-truth translation error"),
        Line2D([], [], color="tab:red", label="Ground-truth rotation error"),
    )
    figure.legend(handles=handles, loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output / "tracking_errors_comparison.png", dpi=180)
    plt.close(figure)


def pixel_errors(root, output):
    datasets = [_evaluation_samples(root, method) for method in METHODS]
    fields = ["pbvs_pixel_error_px"]
    if all("ground_truth_pixel_error_px" in data.dtype.names for data in datasets):
        fields.append("ground_truth_pixel_error_px")
    maximum = max(np.nanmax(np.concatenate([data[field] for field in fields]))
                  for data in datasets) * 1.05
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False, sharey=True)
    for axis, method, data in zip(axes.flat, METHODS, datasets):
        time = data["time"] - data["time"][0]
        axis.plot(time, data["pbvs_pixel_error_px"], color="tab:green",
                  linewidth=1.3)
        if "ground_truth_pixel_error_px" in data.dtype.names:
            axis.plot(time, data["ground_truth_pixel_error_px"], color="tab:blue",
                      linewidth=1.2)
        axis.set(xlabel="Simulation time [s]", ylim=(0, maximum))
        if axis in axes[:, 0]:
            axis.set_ylabel("Pixel error [px]")
        axis.grid(True, alpha=0.25)
        _method_label(axis, method)
    handles = [Line2D([], [], color="tab:green", label="Visual pixel error")]
    if len(fields) == 2:
        handles.append(Line2D([], [], color="tab:blue",
                              label="Ground-truth pixel error"))
    figure.legend(handles=handles, loc="lower center", ncol=len(handles),
                  bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output / "pixel_errors_comparison.png", dpi=180)
    plt.close(figure)


def energies(root, output):
    loaded = []
    maximum = 0.0
    for method in METHODS:
        directory, suffix = _paths(root, method)
        data = _load_csv(directory / f"energy_samples_{suffix}.csv", text=True)
        loaded.append((method, data))
        maximum = max(maximum, float(np.nanmax(np.r_[
            data["ur5_cumulative_energy_j"], data["franka_cumulative_energy_j"]
        ])))
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    stage_colors = plt.get_cmap("Pastel1")
    for axis, (method, data) in zip(axes.flat, loaded):
        time = data["time"] - data["time"][0]
        axis.plot(time, data["ur5_cumulative_energy_j"], color="tab:blue",
                  linewidth=1.4)
        axis.plot(time, data["franka_cumulative_energy_j"], color="tab:orange",
                  linewidth=1.4)
        stages = np.asarray(data["stage"])
        boundaries = np.r_[0, np.flatnonzero(stages[1:] != stages[:-1]) + 1,
                           len(stages)]
        for stage_index, (start, stop) in enumerate(
                zip(boundaries[:-1], boundaries[1:])):
            left, right = time[start], time[stop - 1]
            axis.axvspan(left, right,
                         color=stage_colors(stage_index % 9), alpha=0.18,
                         linewidth=0)
            if right - left > 0.15:
                axis.text((left + right) / 2, 0.02,
                          str(stages[start]).replace("_", " "),
                          transform=axis.get_xaxis_transform(),
                          ha="center", va="bottom", rotation=90, fontsize=9.5)
        axis.set(xlabel="Simulation time [s]", ylim=(0, maximum * 1.05))
        if axis in axes[:, 0]:
            axis.set_ylabel("Cumulative energy [J]")
        axis.grid(True, alpha=0.25)
        _method_label(axis, method)
    handles = (
        Line2D([], [], color="tab:blue", label="UR5 cumulative energy"),
        Line2D([], [], color="tab:orange", label="Franka cumulative energy"),
    )
    figure.legend(handles=handles, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output / "energies_comparison.png", dpi=180)
    plt.close(figure)


def kf_pose_comparison(root, output):
    """Create paired KF-acceleration versus full-pose KF figures."""
    methods = ("kf_acc", "kf_pose")
    datasets = [_evaluation_samples(root, method) for method in methods]

    trajectory_sets = []
    point_sets = []
    for method, samples in zip(methods, datasets):
        directory, suffix = _paths(root, method)
        trajectory = _load_csv(directory / f"trajectory_{suffix}.csv", text=True)
        trajectory_sets.append(trajectory)
        for names, source in (
                (("x", "y", "z"), trajectory),
                (("camera_x", "camera_y", "camera_z"), trajectory),
                (("measured_cube_x", "measured_cube_y", "measured_cube_z"),
                 samples)):
            points = np.column_stack([source[name] for name in names])
            valid = points[np.all(np.isfinite(points), axis=1)]
            if len(valid):
                point_sets.append(valid)
    cloud = np.vstack(point_sets)
    low, high = np.min(cloud, axis=0), np.max(cloud, axis=0)
    padding = np.maximum(0.03, 0.05 * (high - low))
    figure = plt.figure(figsize=(15, 7))
    for index, (method, trajectory, samples) in enumerate(
            zip(methods, trajectory_sets, datasets), 1):
        axis = figure.add_subplot(1, 2, index, projection="3d")
        axis.plot(trajectory["x"], trajectory["y"], trajectory["z"],
                  color="tab:blue", linewidth=1.5)
        valid = np.isfinite(samples["measured_cube_x"])
        axis.plot(samples["measured_cube_x"][valid],
                  samples["measured_cube_y"][valid],
                  samples["measured_cube_z"][valid], color="black",
                  linestyle="none", marker=".", markersize=2.5)
        axis.plot(trajectory["camera_x"], trajectory["camera_y"],
                  trajectory["camera_z"], color="tab:purple",
                  linewidth=1.2, alpha=0.8)
        axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
        axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
        axis.set_zlim(low[2] - padding[2], high[2] + padding[2])
        _method_label(axis, method)
    figure.legend(handles=(
        Line2D([], [], color="tab:blue", label="Ground-truth cube"),
        Line2D([], [], color="black", marker=".", linestyle="none",
               label="Camera-measured cube"),
        Line2D([], [], color="tab:purple", label="Franka camera"),
    ), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(output / "trajectories_kf_acc_vs_kf_pose.png", dpi=180)
    plt.close(figure)

    translation_max = 1.05 * max(
        np.nanmax(np.r_[data["pbvs_translation_error_m"],
                        data["ground_truth_tracking_error_m"]])
        for data in datasets
    ) * 1000
    rotation_max = 1.05 * max(
        np.nanmax(np.degrees(data["ground_truth_rotation_error_rad"]))
        for data in datasets
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for axis, method, data in zip(axes, methods, datasets):
        time = data["time"] - data["time"][0]
        axis.plot(time, 1000 * data["pbvs_translation_error_m"],
                  color="tab:blue", linewidth=1.3)
        axis.plot(time, 1000 * data["ground_truth_tracking_error_m"],
                  color="tab:orange", linewidth=1.2)
        axis.set(xlabel="Simulation time [s]", ylim=(0, translation_max))
        axis.grid(True, alpha=0.25)
        rotation_axis = axis.twinx()
        rotation_axis.plot(time, np.degrees(data["ground_truth_rotation_error_rad"]),
                           color="tab:red", linewidth=1.1)
        rotation_axis.set_ylim(0, rotation_max)
        if axis is axes[-1]:
            rotation_axis.set_ylabel("Rotation error [deg]")
        else:
            rotation_axis.set_yticklabels([])
        _method_label(axis, method)
    axes[0].set_ylabel("Translation error [mm]")
    figure.legend(handles=(
        Line2D([], [], color="tab:blue", label="Visual translation error"),
        Line2D([], [], color="tab:orange", label="Ground-truth translation error"),
        Line2D([], [], color="tab:red", label="Ground-truth rotation error"),
    ), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output / "tracking_errors_kf_acc_vs_kf_pose.png", dpi=180)
    plt.close(figure)

    pixel_max = 1.05 * max(
        np.nanmax(np.r_[data["pbvs_pixel_error_px"],
                        data["ground_truth_pixel_error_px"]])
        for data in datasets
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, method, data in zip(axes, methods, datasets):
        time = data["time"] - data["time"][0]
        axis.plot(time, data["pbvs_pixel_error_px"], color="tab:green",
                  linewidth=1.3)
        axis.plot(time, data["ground_truth_pixel_error_px"], color="tab:blue",
                  linewidth=1.2)
        axis.set(xlabel="Simulation time [s]", ylim=(0, pixel_max))
        axis.grid(True, alpha=0.25)
        _method_label(axis, method)
    axes[0].set_ylabel("Pixel error [px]")
    figure.legend(handles=(
        Line2D([], [], color="tab:green", label="Visual pixel error"),
        Line2D([], [], color="tab:blue", label="Ground-truth pixel error"),
    ), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(output / "pixel_errors_kf_acc_vs_kf_pose.png", dpi=180)
    plt.close(figure)

    energy_sets = []
    energy_max = 0.0
    for method in methods:
        directory, suffix = _paths(root, method)
        data = _load_csv(directory / f"energy_samples_{suffix}.csv", text=True)
        energy_sets.append(data)
        energy_max = max(energy_max, np.nanmax(np.r_[
            data["ur5_cumulative_energy_j"], data["franka_cumulative_energy_j"]
        ]))
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    stage_colors = plt.get_cmap("Pastel1")
    for axis, method, data in zip(axes, methods, energy_sets):
        time = data["time"] - data["time"][0]
        axis.plot(time, data["ur5_cumulative_energy_j"], color="tab:blue",
                  linewidth=1.4)
        axis.plot(time, data["franka_cumulative_energy_j"], color="tab:orange",
                  linewidth=1.4)
        stages = np.asarray(data["stage"])
        boundaries = np.r_[0, np.flatnonzero(stages[1:] != stages[:-1]) + 1,
                           len(stages)]
        for stage_index, (start, stop) in enumerate(zip(boundaries[:-1],
                                                        boundaries[1:])):
            left, right = time[start], time[stop - 1]
            axis.axvspan(left, right, color=stage_colors(stage_index % 9),
                         alpha=0.18, linewidth=0)
            if right - left > 0.15:
                axis.text((left + right) / 2, 0.02,
                          str(stages[start]).replace("_", " "),
                          transform=axis.get_xaxis_transform(), ha="center",
                          va="bottom", rotation=90, fontsize=9.5)
        axis.set(xlabel="Simulation time [s]", ylim=(0, energy_max * 1.05))
        axis.grid(True, alpha=0.25)
        _method_label(axis, method)
    axes[0].set_ylabel("Cumulative energy [J]")
    figure.legend(handles=(
        Line2D([], [], color="tab:blue", label="UR5 cumulative energy"),
        Line2D([], [], color="tab:orange", label="Franka cumulative energy"),
    ), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(output / "energies_kf_acc_vs_kf_pose.png", dpi=180)
    plt.close(figure)


def generate(runs_directory="metrics/runs"):
    root = Path(runs_directory)
    output = root.parent / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    trajectories(root, output)
    tracking_errors(root, output)
    pixel_errors(root, output)
    energies(root, output)
    paired_methods = ("kf_acc", "kf_pose")
    if all((root / f"run_{method}" / f"samples_{method}.csv").exists()
           and (root / f"run_{method}" / f"energy_samples_{method}.csv").exists()
           for method in paired_methods):
        kf_pose_comparison(root, output)
    print(f"Saved combined comparison figures to {output}")
    return output


if __name__ == "__main__":
    generate()
