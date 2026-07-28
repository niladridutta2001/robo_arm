"""Plot desired, measured, and controlled trajectories from the latest run."""

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


def generate(run_directory="metrics/runs/run_ik", show=True):
    run_directory = Path(run_directory)
    candidates = sorted(run_directory.glob("trajectory_*.csv"))
    path = candidates[0] if candidates else run_directory / "trajectory.csv"
    if not path.exists():
        raise SystemExit("Run `python main.py --direct --fast` first.")

    data = np.atleast_1d(np.genfromtxt(
        path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    ))
    suffix = run_directory.name.removeprefix("run_")
    metrics_path = run_directory / f"samples_{suffix}.csv"
    measurements = (np.atleast_1d(np.genfromtxt(metrics_path, delimiter=",", names=True))
                    if metrics_path.exists() else None)

    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(111, projection="3d")
    colors = {"circle": "tab:blue", "lissajous": "tab:red"}

    for pattern in np.unique(data["pattern"]):
        rows = data[data["pattern"] == pattern]
        color = colors.get(pattern, "black")
        axis.plot(rows["x"], rows["y"], rows["z"], color=color,
                  label=f"{pattern} ground truth")
        axis.plot(rows["camera_x"], rows["camera_y"], rows["camera_z"],
                  color=color, alpha=0.45, linewidth=1.2,
                  label=f"camera during {pattern}")

    if measurements is not None and "measured_cube_x" in measurements.dtype.names:
        valid = np.isfinite(measurements["measured_cube_x"])
        axis.plot(measurements["measured_cube_x"][valid],
                  measurements["measured_cube_y"][valid],
                  measurements["measured_cube_z"][valid],
                  color="black", linestyle="none", marker=".", markersize=2.5,
                  label="camera-measured cube")

    is_kalman_run = any(name in run_directory.name
                        for name in ("kf_vel", "kf_acc", "kf_pose"))
    if (measurements is not None
            and "filtered_cube_x" in measurements.dtype.names
            and not is_kalman_run):
        valid = np.isfinite(measurements["filtered_cube_x"])
        if np.any(valid):
            if "nn_motion" in run_directory.name:
                filter_label = "Neural motion estimate"
            elif "rls" in run_directory.name:
                filter_label = "RLS Fourier estimate"
            else:
                filter_label = "Filtered cube"
            axis.plot(measurements["filtered_cube_x"][valid],
                      measurements["filtered_cube_y"][valid],
                      measurements["filtered_cube_z"][valid],
                      color="tab:orange", linewidth=1.4,
                      label=filter_label)

    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    axis.set_xlim(-0.65, 0.40)
    axis.set_ylim(-0.40, 0.25)
    axis.set_zlim(0.85, 1.40)
    axis.set_box_aspect((1.05, 0.65, 0.55))
    fig.tight_layout()
    output = run_directory / f"trajectory_{suffix}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main():
    generate(show=True)


if __name__ == "__main__":
    main()
