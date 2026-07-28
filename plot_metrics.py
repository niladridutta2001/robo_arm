"""Plot synchronization errors saved by metrics.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 19.2,
    "axes.labelsize": 16,
    "xtick.labelsize": 12.8,
    "ytick.labelsize": 12.8,
    "legend.fontsize": 16,
    "figure.titlesize": 19.2,
})


def _shared_limits(runs_directory):
    translation, rotation, pixels = [], [], []
    for run_directory in Path(runs_directory).glob("run_*"):
        suffix = run_directory.name.removeprefix("run_")
        sample_path = run_directory / f"samples_{suffix}.csv"
        if not sample_path.exists():
            continue
        values = np.atleast_1d(np.genfromtxt(sample_path, delimiter=",", names=True))
        if "evaluation_active" in values.dtype.names:
            values = values[values["evaluation_active"] > 0.5]
        translation.extend((1000 * values["pbvs_translation_error_m"],
                            1000 * values["ground_truth_tracking_error_m"]))
        rotation.append(np.degrees(values["ground_truth_rotation_error_rad"]))
        if "visual_rotation_error_rad" in values.dtype.names:
            rotation.append(np.degrees(values["visual_rotation_error_rad"]))
        pixels.append(values["pbvs_pixel_error_px"])
        if "ground_truth_pixel_error_px" in values.dtype.names:
            pixels.append(values["ground_truth_pixel_error_px"])

    def upper(arrays, minimum):
        valid = [a[np.isfinite(a)] for a in arrays if np.size(a)]
        if not valid:
            return minimum
        finite = np.concatenate(valid)
        return max(minimum, 1.05 * float(np.max(finite))) if finite.size else minimum

    return upper(translation, 50), upper(rotation, 5), upper(pixels, 25)


def generate(run_directory="metrics/runs/run_ik", show=False):
    run_directory = Path(run_directory)
    suffix = run_directory.name.removeprefix("run_")
    path = run_directory / f"samples_{suffix}.csv"
    if not path.exists():
        raise SystemExit("Run `python main.py` first to generate metrics.")

    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if "evaluation_active" in data.dtype.names:
        data = data[data["evaluation_active"] > 0.5]
    time = data["time"] - data["time"][0]
    translation_max, rotation_max, pixel_max = _shared_limits(path.parent.parent)

    fig, translation_axis = plt.subplots(figsize=(10, 5.5))
    translation_axis.plot(time, 1000 * data["pbvs_translation_error_m"],
                          label="Visual translation error", linewidth=1.4)
    translation_axis.plot(time, 1000 * data["ground_truth_tracking_error_m"],
                          label="Ground-truth translation error", linewidth=1.2)
    translation_axis.set(xlabel="Simulation time [s]",
                         ylabel="Translation error [mm]")
    translation_axis.set_ylim(0, translation_max)
    translation_axis.grid(True, alpha=0.3)

    rotation_axis = translation_axis.twinx()
    rotation_axis.plot(time, np.degrees(data["ground_truth_rotation_error_rad"]),
                       color="tab:red", label="Ground-truth rotation error", alpha=0.85)
    if ("visual_rotation_error_rad" in data.dtype.names and
            np.any(np.isfinite(data["visual_rotation_error_rad"]))):
        rotation_axis.plot(time, np.degrees(data["visual_rotation_error_rad"]),
                           color="tab:purple", linestyle="--",
                           label="Visual rotation error", alpha=0.85)
    rotation_axis.set_ylabel("Rotation error [deg]")
    rotation_axis.set_ylim(0, rotation_max)
    fig.tight_layout()
    tracking_output = path.parent / f"tracking_errors_{suffix}.png"
    fig.savefig(tracking_output, dpi=180)

    pixel_fig, pixel_axis = plt.subplots(figsize=(10, 4))
    pixel_axis.plot(time, data["pbvs_pixel_error_px"], color="tab:green",
                    label="Visual pixel error", linewidth=1.4)
    if "ground_truth_pixel_error_px" in data.dtype.names:
        pixel_axis.plot(time, data["ground_truth_pixel_error_px"],
                        color="tab:blue", label="Ground-truth pixel error",
                        linewidth=1.2)
    pixel_axis.set(xlabel="Simulation time [s]", ylabel="Pixel error [px]")
    pixel_axis.set_ylim(0, pixel_max)
    pixel_axis.grid(True, alpha=0.3)
    pixel_fig.tight_layout()
    pixel_output = path.parent / f"pixel_error_{suffix}.png"
    pixel_fig.savefig(pixel_output, dpi=180)

    pose_fig = None
    pose_output = path.parent / f"raw_pnp_validation_{suffix}.png"
    if ("raw_pnp_rotation_gt_error_rad" in data.dtype.names and
            np.any(np.isfinite(data["raw_pnp_rotation_gt_error_rad"]))):
        pose_fig, pose_axis = plt.subplots(figsize=(10, 4))
        pose_axis.plot(
            time, np.degrees(data["raw_pnp_rotation_gt_error_rad"]),
            color="tab:orange", linewidth=1.4,
        )
        pose_axis.set(xlabel="Simulation time [s]",
                      ylabel="Raw PnP-to-GT rotation error [deg]")
        pose_axis.grid(True, alpha=0.3)
        pose_fig.tight_layout()
        pose_fig.savefig(pose_output, dpi=180)

    energy_fig = None
    energy_output = path.parent / f"energy_{suffix}.png"
    energy_path = path.parent / f"energy_samples_{suffix}.csv"
    if energy_path.exists():
        energy_data = np.atleast_1d(np.genfromtxt(
            energy_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        ))
        energy_time = energy_data["time"] - energy_data["time"][0]
        energy_fig, energy_axis = plt.subplots(figsize=(10, 4.5))
        energy_axis.plot(energy_time, energy_data["ur5_cumulative_energy_j"],
                         color="tab:blue", label="UR5 cumulative energy",
                         linewidth=1.5)
        energy_axis.plot(energy_time, energy_data["franka_cumulative_energy_j"],
                         color="tab:orange", label="Franka cumulative energy",
                         linewidth=1.5)
        stages = np.asarray(energy_data["stage"])
        boundaries = np.r_[0, np.flatnonzero(stages[1:] != stages[:-1]) + 1,
                           len(stages)]
        stage_colors = plt.get_cmap("Pastel1")
        for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            left = energy_time[start]
            right = energy_time[stop - 1]
            energy_axis.axvspan(left, right, color=stage_colors(index % 9),
                                alpha=0.22, linewidth=0)
            if right - left > 0.18:
                energy_axis.text((left + right) / 2, 0.98, str(stages[start]),
                                 transform=energy_axis.get_xaxis_transform(),
                                 ha="center", va="top", rotation=90, fontsize=11.2)
        energy_axis.set(xlabel="Simulation time [s]",
                        ylabel="Cumulative energy [J]")
        energy_axis.grid(True, alpha=0.3)
        energy_fig.tight_layout()
        energy_fig.savefig(energy_output, dpi=180)

    print(f"Saved {tracking_output}")
    print(f"Saved {pixel_output}")
    if pose_fig is not None:
        print(f"Saved {pose_output}")
    if energy_fig is not None:
        print(f"Saved {energy_output}")
    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(pixel_fig)
        if pose_fig is not None:
            plt.close(pose_fig)
        if energy_fig is not None:
            plt.close(energy_fig)
    return fig, pixel_fig, energy_fig


def main():
    generate(show=False)


if __name__ == "__main__":
    main()
