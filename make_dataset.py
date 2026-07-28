"""Generate varied visual-motion sequences for offline supervised training."""

from pathlib import Path

import numpy as np


RATE, DURATION, SEQUENCES = 30, 20.0, 50


def smoothstep(value):
    return value * value * (3.0 - 2.0 * value)


def main():
    rng = np.random.default_rng(42)
    time = np.arange(0, DURATION, 1.0 / RATE)
    measured_all, truth_all = [], []

    for sequence in range(SEQUENCES):
        center = np.array((-0.18, 0.0, 1.05)) + rng.uniform(
            (-0.04, -0.04, -0.05), (0.04, 0.04, 0.05)
        )
        radius = rng.uniform(0.08, 0.15)
        depth = rng.uniform(0.02, 0.06)
        period = rng.uniform(3.0, 8.0)
        phase_offset = rng.uniform(0, 2 * np.pi)
        truth = np.empty((len(time), 3))
        for index, current_time in enumerate(time):
            local = (current_time % period) / period
            phase = 2 * np.pi * smoothstep(local) + phase_offset
            pattern = sequence % 2
            truth[index, 0] = center[0] + depth * np.sin(phase)
            if pattern == 0:
                truth[index, 1] = center[1] + radius * np.cos(phase)
                truth[index, 2] = center[2] + radius * np.sin(phase)
            else:
                truth[index, 1] = center[1] + radius * np.cos(phase)
                truth[index, 2] = center[2] + 0.75 * radius * np.sin(2 * phase)

        measured = truth + rng.normal(0, (0.004, 0.004, 0.008), truth.shape)
        # Random single-frame and short-burst visual losses.
        missing = rng.random(len(time)) < 0.02
        for _ in range(rng.integers(1, 4)):
            start = rng.integers(0, len(time) - 8)
            missing[start:start + rng.integers(2, 8)] = True
        measured[missing] = np.nan
        measured_all.append(measured)
        truth_all.append(truth)

    truth = np.asarray(truth_all)
    measured = np.asarray(measured_all)
    velocity = np.gradient(truth, 1.0 / RATE, axis=1)
    acceleration = np.gradient(velocity, 1.0 / RATE, axis=1)
    output = Path("datasets/motion_dataset.npz")
    output.parent.mkdir(exist_ok=True)
    np.savez_compressed(output, measured=measured, position=truth,
                        velocity=velocity, acceleration=acceleration,
                        rate=RATE)
    valid = int(np.isfinite(measured[..., 0]).sum())
    print(f"Saved {output}: {SEQUENCES} sequences, {truth.size // 3} frames, "
          f"{valid} valid visual measurements")


if __name__ == "__main__":
    main()
