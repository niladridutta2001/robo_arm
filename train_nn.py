"""Train the PyTorch motion MLP with offline simulator ground truth."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from motion_network import MotionMLP


WINDOW, EPOCHS, BATCH_SIZE = 15, 150, 512


def windows(dataset, sequence_indices):
    inputs, targets = [], []
    for sequence in sequence_indices:
        measured = dataset["measured"][sequence]
        target = np.concatenate((dataset["position"][sequence],
                                 dataset["velocity"][sequence],
                                 dataset["acceleration"][sequence]), axis=1)
        for index in range(WINDOW - 1, len(measured)):
            history = measured[index - WINDOW + 1:index + 1]
            if np.all(np.isfinite(history)):
                inputs.append(history.reshape(-1))
                targets.append(target[index])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is False.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    path = Path("datasets/motion_dataset.npz")
    if not path.exists():
        raise SystemExit("Run `python make_dataset.py` first.")
    dataset = np.load(path)
    split = int(0.8 * len(dataset["measured"]))
    x_train, y_train = windows(dataset, range(split))
    x_valid, y_valid = windows(dataset, range(split, len(dataset["measured"])))
    print(f"Training windows: {len(x_train)}, validation windows: {len(x_valid)}")

    x_mean, x_std = x_train.mean(0), x_train.std(0) + 1e-6
    y_mean, y_std = y_train.mean(0), y_train.std(0) + 1e-6
    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    x_valid = (x_valid - x_mean) / x_std
    y_valid_normalized = (y_valid - y_mean) / y_std

    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=device.type == "cuda")
    model = MotionMLP(WINDOW).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()
    valid_x = torch.from_numpy(x_valid).to(device)
    valid_y = torch.from_numpy(y_valid_normalized).to(device)
    best_loss, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(valid_x), valid_y).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch {epoch:3d}: validation MSE={validation_loss:.5f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(valid_x).cpu().numpy() * y_std + y_mean
    rmse = np.sqrt(np.mean((prediction - y_valid) ** 2, axis=0))
    print("Validation RMSE: position=%.4f m, velocity=%.4f m/s, acceleration=%.4f m/s^2"
          % (np.mean(rmse[:3]), np.mean(rmse[3:6]), np.mean(rmse[6:9])))

    output = Path("models/motion_mlp.pt")
    output.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "window": WINDOW,
                "x_mean": x_mean, "x_std": x_std,
                "y_mean": y_mean, "y_std": y_std}, output)
    print(f"Saved {output} using {device}")


if __name__ == "__main__":
    main()
