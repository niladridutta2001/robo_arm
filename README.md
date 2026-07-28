# Visual-guided dual-arm synchronization

Minimal PyBullet implementation of the assignment. The current milestone sets up a UR5
with a Robotiq 85 gripper opposite a 7-DoF Franka Panda. Robot A picks up a marked cube,
transfers it into the shared workspace, and alternates between a circle and a Lissajous
curve in the vertical `yz` plane while adding sinusoidal motion along `x`.

Robot B is intentionally stationary at this stage. Its visual-only tracking controller
will be added as the next milestone; no Robot A state is passed to it.

## Run

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

For a quick headless check:

```powershell
.venv\Scripts\python main.py --direct --fast --methods ik
```

Multiple methods can be run sequentially in one command, for example:

```powershell
.venv\Scripts\python main.py --direct --fast --methods ik kf_vel kf_acc
```

The default motion executes two alternating pairs: circle, Lissajous, circle,
Lissajous. Use `--cycles 1` for the shorter single-pair experiment.

After generating TOPP schedules with `time_optimal.py`, execute the same Franka
synchronization experiments against time-optimal Robot A motion with:

```powershell
.venv\Scripts\python main.py --direct --fast --robot-a-motion topp --methods ik kf_acc
```


Use `--methods kf_vel` for constant-velocity Kalman filtering or
`--methods kf_acc` for constant-acceleration filtering. Results are stored in
separate method-named directories under `metrics/runs`.
Use `--methods kf_pose` for marker-based translational and rotational
constant-acceleration Kalman filtering and 6-D relative-pose regulation.
Use `--methods nn_online` for camera-only one-step neural adaptation. It starts
from the offline checkpoint and updates only the final layer during execution.
Use `--methods rls` for FFT-initialized online Fourier RLS prediction.

Plan Robot A's prescribed paths separately with joint-constrained TOPP:

```powershell
.venv\Scripts\python time_optimal.py
```

Train and evaluate the experimental motion MLP with:

```powershell
.venv\Scripts\python make_dataset.py
.venv\Scripts\python train_nn.py
.venv\Scripts\python main.py --direct --fast --methods nn_motion
```
