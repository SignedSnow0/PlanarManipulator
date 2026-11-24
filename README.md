# Planar Manipulator

A small Python/Matplotlib application that visualises the workspace of a
joint-limited **two-link** planar robot arm and animates it to a clicked target.

## Install

```powershell
pip install -r requirements.txt
```

Dependencies:

- `matplotlib` — plotting, animation, and mouse events
- `numpy` — trigonometry, angle interpolation, and clipping
- `shapely` — workspace boundary construction, polygonising, and reachability geometry

## Run

```powershell
python main.py
```

A figure appears with the reachable workspace in white and unreachable space
in light gray. The current pose is drawn as two segments: the first link in
blue and the second in orange.

- **Left click** inside the workspace: move to the target using the *elbow up*
  inverse-kinematics solution.
- **Right click** inside the workspace: move to the target using the *elbow
  down* solution.
- Clicking outside the reachable area prints a message and leaves the arm
  unchanged.

The target, solved angles (in degrees), and forward-kinematics error are
printed to the console for each click.

## Files

- `main.py` — the `TwoLinkArm` class, configuration, mouse handling, and animation.
- `workspace.py` — workspace geometry and plotting.
- `requirements.txt` — Python dependencies.

## How it works

The robot-specific code lives in the `TwoLinkArm` class:

| Method | Purpose |
| --- | --- |
| `joint()` | Elbow position from the first angle. |
| `endpoint()` | Tip position from both absolute angles. |
| `is_reachable()` | Law-of-cosines joint-limit feasibility test. |
| `inverse()` | Target point to absolute angles, choosing an elbow branch. |
| `draw()` / `move()` | Create and update the blue and orange link lines. |
| `animate()` | Interpolate to a new pose and show the animation. |

`workspace.py` keeps the Shapely/Matplotlib workspace code separate:

| Function | Purpose |
| --- | --- |
| `build_workspace()` | Boundary circles, polygonised into reachable faces. |
| `plot_workspace()` | Draw reachable faces and unreachable holes. |

`build_workspace()` takes circles from the joint limits and link lengths,
adds full-extension and maximum-fold circles when those relative angles are
allowed, nodes them with `unary_union`, splits them with `polygonize`, and
keeps faces whose representative point passes `arm.is_reachable()`.

`on_click()` handles mouse events, checks reachability, solves inverse
kinematics, prints the result, and starts the animation. The animation object
is retained on the `arm` instance so it is not garbage-collected mid-run.

## Angle conventions

- `theta1` and `theta2` are **absolute** angles in radians, measured from the
  global x-axis.
- `limits1` applies to `theta1`.
- `limits2` applies to the angle of the second link **relative** to the first
  link, matching the law-of-cosines derivation.
- The base is at `(0, 0)`; lengths are configured in the `arm` instantiation
  near the bottom of `main.py`.
