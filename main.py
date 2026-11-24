"""Interactive visualisation and inverse kinematics for a two-link planar arm.

The script draws the joint-limited workspace of a two-link manipulator and its
current pose. Left-click a target for the "elbow up" inverse-kinematics
solution, or right-click for the "elbow down" solution; the arm animates to
that pose. Workspace construction and plotting live in ``workspace.py``.

Dependencies are listed in ``requirements.txt``. Run with::

    python main.py
"""

import matplotlib.pyplot as plot
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseButton
import numpy as np

from workspace import plot_workspace


class TwoLinkArm:
    """A planar arm made of exactly two links.

    Link angles are absolute angles in radians measured from the global
    x-axis. Joint limits are stored as ``(lower, upper)`` pairs: the first
    link's limits apply to its absolute angle, while the second link's limits
    apply to the angle *relative* to the first link. The base is at the origin.

    Attributes:
        l1: Length of the first link.
        l2: Length of the second link.
        theta1: Current absolute angle of the first link, in radians.
        theta2: Current absolute angle of the second link, in radians.
        limits1: ``(min, max)`` absolute joint limits for the first link.
        limits2: ``(min, max)`` relative joint limits for the second link.
        lines: The two Matplotlib artists created by :meth:`draw` and
            updated by :meth:`move`; one per link.
        _animation: Most recent animation, kept alive so it is not collected.
    """

    def __init__(self, l1, l2, theta1, theta2, limits1, limits2):
        """Create a two-link arm with lengths, pose, and joint limits."""
        self.l1 = l1
        self.l2 = l2
        self.theta1 = theta1
        self.theta2 = theta2
        self.limits1 = limits1
        self.limits2 = limits2
        self.lines = []
        self._animation = None

    def joint(self, theta1=None):
        """Return the elbow position for ``theta1`` (defaults to current)."""
        if theta1 is None:
            theta1 = self.theta1
        return (self.l1 * np.cos(theta1), self.l1 * np.sin(theta1))

    def endpoint(self, theta1=None, theta2=None):
        """Return the tip position for absolute angles (defaults to current)."""
        if theta1 is None:
            theta1 = self.theta1
        if theta2 is None:
            theta2 = self.theta2
        return (
            self.l1 * np.cos(theta1) + self.l2 * np.cos(theta2),
            self.l1 * np.sin(theta1) + self.l2 * np.sin(theta2),
        )

    def is_reachable(self, x, y, tol=1e-7):
        """Return whether ``(x, y)`` satisfies both joint limits.

        The law of cosines gives the two possible relative angles for the
        second joint. Each is checked against ``limits2``, and the resulting
        first-joint angle is checked against ``limits1``.
        """
        cos_q2 = (x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if abs(cos_q2) > 1.0 + tol:
            return False
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)

        q2_min, q2_max = self.limits2
        q1_min, q1_max = self.limits1

        for sign in (1.0, -1.0):
            q2 = sign * np.arccos(cos_q2)
            if not (q2_min - tol <= q2 <= q2_max + tol):
                continue

            q1 = np.arctan2(y, x) - np.arctan2(
                self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2)
            )
            while q1 < q1_min - np.pi:
                q1 += 2 * np.pi
            while q1 > q1_max + np.pi:
                q1 -= 2 * np.pi

            if q1_min - tol <= q1 <= q1_max + tol:
                return True
        return False

    def inverse(self, x, y, elbow_up=True):
        """Return absolute angles reaching ``(x, y)``.

        ``elbow_up=True`` selects the negative relative angle for the second
        joint; ``False`` selects the positive one. The caller should check
        :meth:`is_reachable` first, since an unreachable target can produce
        ``nan`` from ``arccos``.
        """
        q2 = np.arccos((x**2 + y**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2))
        if elbow_up:
            q2 = -q2

        q1 = np.arctan2(y, x) - np.arctan2(
            self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2)
        )
        return q1, q1 + q2

    def draw(self, ax):
        """Draw the first link in blue and the second in orange."""
        elbow = self.joint()
        tip = self.endpoint()
        self.lines = [
            *ax.plot([0, elbow[0]], [0, elbow[1]], 'o-', lw=2, color='tab:blue'),
            *ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], 'o-', lw=2, color='tab:orange'),
        ]

    def move(self, theta1, theta2):
        """Set a new absolute pose and update both drawn links in place."""
        self.theta1, self.theta2 = theta1, theta2
        elbow = self.joint()
        tip = self.endpoint()
        self.lines[0].set_data([0, elbow[0]], [0, elbow[1]])
        self.lines[1].set_data([elbow[0], tip[0]], [elbow[1], tip[1]])

    def animate(self, fig, to_angles, frames=100):
        """Interpolate from the current pose to ``to_angles`` and display it."""
        angles1 = np.linspace(self.theta1, to_angles[0], frames)
        angles2 = np.linspace(self.theta2, to_angles[1], frames)

        def update(frame):
            self.move(angles1[frame], angles2[frame])
            return self.lines

        self._animation = FuncAnimation(
            fig, update, frames=frames, interval=20, blit=True, repeat=False
        )
        plot.show()


# Two-link configuration: lengths, initial absolute angles, and joint limits.
arm = TwoLinkArm(
    l1=7,
    l2=3,
    theta1=np.pi / 4,
    theta2=np.pi / 3,
    limits1=(-np.pi / 2, np.pi),
    limits2=(-np.pi, np.pi / 2),
)

fig, ax = plot.subplots(figsize=(6, 6))
plot_workspace(ax, arm)
arm.draw(ax)


def on_click(event):
    """Move the arm to a clicked target using the selected elbow branch."""
    if event.button not in (MouseButton.LEFT, MouseButton.RIGHT):
        return
    if event.xdata is None or event.ydata is None:
        return

    x, y = event.xdata, event.ydata
    if not arm.is_reachable(x, y):
        print(f"Target: ({x:.2f}, {y:.2f}) is not reachable.")
        return

    theta1, theta2 = arm.inverse(x, y, elbow_up=event.button == MouseButton.LEFT)
    px, py = arm.endpoint(theta1, theta2)
    print(f"Target: ({x:.2f}, {y:.2f})")
    print(f"Calculated angles: theta1 = {np.degrees(theta1):.2f} degrees, theta2 = {np.degrees(theta2):.2f} degrees")
    print(f"Error: ({abs(x - px)}, {abs(y - py)})")

    arm.animate(fig, (theta1, theta2), frames=100)


fig.canvas.mpl_connect('button_press_event', on_click)
plot.show()
