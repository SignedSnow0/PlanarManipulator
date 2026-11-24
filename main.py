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
from matplotlib.widgets import CheckButtons
import numpy as np

from workspace import plot_workspace_2, plot_workspace_3


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

    def forward(self, theta1=None, theta2=None):
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
        tip = self.forward()
        self.lines = [
            *ax.plot([0, elbow[0]], [0, elbow[1]], 'o-', lw=2, color='tab:blue'),
            *ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], 'o-', lw=2, color='tab:orange'),
        ]

    def move(self, theta1, theta2):
        """Set a new absolute pose and update both drawn links in place."""
        self.theta1, self.theta2 = theta1, theta2
        elbow = self.joint()
        tip = self.forward()
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

class ThreeLinkArm:
    def __init__(self, l1, l2, l3, theta1, theta2, theta3, limits1, limits2, limits3):
        """Create a three-link arm with lengths, pose, and joint limits."""
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.limits1 = limits1
        self.limits2 = limits2
        self.limits3 = limits3
        self.lines = []
        self._animation = None

    def forward(self, theta1=None, theta2=None, theta3=None):
        """Return the tip position for absolute angles (defaults to current)."""
        if theta1 is None:
            theta1 = self.theta1
        if theta2 is None:
            theta2 = self.theta2
        if theta3 is None:
            theta3 = self.theta3

        x = (self.l1 * np.cos(theta1) +
             self.l2 * np.cos(theta2) +
             self.l3 * np.cos(theta3))
        y = (self.l1 * np.sin(theta1) +
             self.l2 * np.sin(theta2) +
             self.l3 * np.sin(theta3))
        
        return x, y

    def inverse(self, x, y, max_iters=500, tol=1e-6):
        if not self.is_reachable(x, y):
            return self.theta1, self.theta2, self.theta3

        theta = np.array([self.theta1, self.theta2, self.theta3], dtype=float)
        learning_rate = 0.2
        damping_factor = 1e-4

        for _ in range(max_iters):
            x_t, y_t = self.forward(theta[0], theta[1], theta[2])
            error = np.array([x - x_t, y - y_t])

            if np.linalg.norm(error) < tol:
                return theta[0], theta[1], theta[2]

            # Global-angle Jacobian: each column depends only on its own global theta_i
            J = np.array([
                [-self.l1 * np.sin(theta[0]), -self.l2 * np.sin(theta[1]), -self.l3 * np.sin(theta[2])],
                [ self.l1 * np.cos(theta[0]),  self.l2 * np.cos(theta[1]),  self.l3 * np.cos(theta[2])],
            ])

            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping_factor * np.eye(2))
            theta += learning_rate * (J_pinv @ error)

            # Enforce relative joint limits sequentially: q1 = t1, q2 = t2 - t1, q3 = t3 - t2
            q1 = np.clip(theta[0], self.limits1[0], self.limits1[1])
            q2 = np.clip(theta[1] - q1, self.limits2[0], self.limits2[1])
            q3 = np.clip(theta[2] - (q1 + q2), self.limits3[0], self.limits3[1])

            theta[0] = q1
            theta[1] = q1 + q2
            theta[2] = q1 + q2 + q3

        return self._cached_ik

    def joint(self, theta1=None, theta2=None):
        """Return the elbow position for ``theta1`` (defaults to current)."""
        if theta1 is None:
            theta1 = self.theta1
        if theta2 is None:
            theta2 = self.theta2
        return (self.l1 * np.cos(theta1) + self.l2 * np.cos(theta2), self.l1 * np.sin(theta1) + self.l2 * np.sin(theta2))

    def joint_elbow(self, theta1=None, theta2=None):
        """Return the elbow position for ``theta1`` (defaults to current)."""
        if theta1 is None:
            theta1 = self.theta1
        return (self.l1 * np.cos(theta1), self.l1 * np.sin(theta1))

    def is_reachable(self, x, y, tol=1e-7):
        r_max = self.l1 + self.l2 + self.l3
        r_min = max(0, self.l1 - self.l2 - self.l3, self.l2 - self.l1 - self.l3, self.l3 - self.l1 - self.l2)

        r = np.sqrt(x**2 + y**2)
        if r < r_min - tol or r > r_max + tol:
            return False

        phi_min = self.limits1[0] + self.limits2[0] + self.limits3[0]
        phi_max = self.limits1[1] + self.limits2[1] + self.limits3[1]
        phis = np.linspace(phi_min, phi_max, abs(int((phi_max - phi_min) / 0.01)) + 1)

        for phi in phis:
            x_w = x - self.l3 * np.cos(phi)
            y_w = y - self.l3 * np.sin(phi)
            r_w_2 = x_w**2 + y_w**2

            if r_w_2 < (self.l1 - self.l2)**2 - tol or r_w_2 > (self.l1 + self.l2)**2 + tol:
                continue

            cos_theta2 = np.clip((r_w_2 - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2), -1.0, 1.0)
            t2_abs = np.arccos(cos_theta2)

            for sign in (1.0, -1.0):
                q2 = wrap_to_interval(sign * t2_abs, self.limits2[0], self.limits2[1], tol)
                if q2 is None:
                    continue

                raw_q1 = np.arctan2(y_w, x_w) - np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
                q1 = wrap_to_interval(raw_q1, self.limits1[0], self.limits1[1], tol)
                if q1 is None:
                    continue

                q3 = phi - q1 - q2
                if self.limits3[0] - tol <= q3 <= self.limits3[1] + tol:
                    # Convert relative (q1, q2, q3) -> global (theta1, theta2, theta3)
                    self._cached_ik = (q1, q1 + q2, q1 + q2 + q3)
                    return True

        return False

    def wrap(self, angle):
        """Wrap an angle to the range [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def draw(self, ax):
        """Draw the first link in blue and the second in orange."""
        elbow_2 = self.joint()
        elbow_1 = self.joint_elbow()
        tip = self.forward()
        self.lines = [
            *ax.plot([0, elbow_1[0]], [0, elbow_1[1]], 'o-', lw=2, color='tab:blue'),
            *ax.plot([elbow_1[0], elbow_2[0]], [elbow_1[1], elbow_2[1]], 'o-', lw=2, color='tab:orange'),
            *ax.plot([elbow_2[0], tip[0]], [elbow_2[1], tip[1]], 'o-', lw=2, color='tab:green'),
        ]

    def move(self, theta1, theta2, theta3):
        """Set a new absolute pose and update both drawn links in place."""
        self.theta1, self.theta2, self.theta3 = theta1, theta2, theta3
        elbow_1 = self.joint_elbow()
        elbow_2 = self.joint()
        tip = self.forward()
        self.lines[0].set_data([0, elbow_1[0]], [0, elbow_1[1]])
        self.lines[1].set_data([elbow_1[0], elbow_2[0]], [elbow_1[1], elbow_2[1]])
        self.lines[2].set_data([elbow_2[0], tip[0]], [elbow_2[1], tip[1]])

    def animate(self, fig, to_angles, frames=100):
        """Interpolate from the current pose to ``to_angles`` and display it."""
        angles1 = np.linspace(self.theta1, to_angles[0], frames)
        angles2 = np.linspace(self.theta2, to_angles[1], frames)
        angles3 = np.linspace(self.theta3, to_angles[2], frames)

        def update(frame):
            self.move(angles1[frame], angles2[frame], angles3[frame])
            return self.lines

        self._animation = FuncAnimation(
            fig, update, frames=frames, interval=20, blit=True, repeat=False
        )
        plot.show()

def wrap_to_interval(angle, low, high, tol=1e-7):
    """Finds angle + 2*pi*k inside [low - tol, high + tol], or returns None."""
    k = np.ceil((low - tol - angle) / (2 * np.pi))
    shifted = angle + k * 2 * np.pi
    if shifted <= high + tol:
        return shifted
    return None
# Two-link configuration: lengths, initial absolute angles, and joint limits.
arm2 = TwoLinkArm(
    l1=7,
    l2=3,
    theta1=np.pi / 4,
    theta2=np.pi / 3,
    limits1=(-np.pi / 2, np.pi),
    limits2=(-np.pi, np.pi / 2),
)

arm3 = ThreeLinkArm(
    l1=7,
    l2=3,
    l3=2,
    theta1=np.pi / 4,
    theta2=np.pi / 3,
    theta3=np.pi / 6,
    limits1=(-np.pi / 2, np.pi),
    limits2=(-np.pi, np.pi / 2),
    limits3=(-np.pi, np.pi / 2),
)

state = { "arm": arm3 }

fig, ax = plot.subplots(figsize=(6, 6))
fig.subplots_adjust(bottom=0.14)

plot_workspace_3(ax, state["arm"])
state["arm"].draw(ax)

check_ax = fig.add_axes([0.35, 0.02, 0.3, 0.07])
check_ax.set_frame_on(False)
check = CheckButtons(check_ax, ['3-link arm'], [True])

def toggle_arm(label):
    """Swap between TwoLinkArm and ThreeLinkArm and redraw."""
    current = state["arm"]
    if current._animation is not None and current._animation.event_source is not None:
        current._animation.event_source.stop()

    is_three_link = check.get_status()[0]
    state["arm"] = arm3 if is_three_link else arm2

    ax.clear()
    if is_three_link:
        plot_workspace_3(ax, state["arm"])
    else:
        plot_workspace_2(ax, state["arm"])
    state["arm"].draw(ax)
    fig.canvas.draw_idle()

def on_click(event):
    """Move the active arm to a clicked target."""
    if event.inaxes is not ax:
        return
    if event.button not in (MouseButton.LEFT, MouseButton.RIGHT):
        return
    if event.xdata is None or event.ydata is None:
        return

    arm = state["arm"]
    x, y = event.xdata, event.ydata

    if not arm.is_reachable(x, y):
        print(f"Target: ({x:.2f}, {y:.2f}) is not reachable.")
        return

    if isinstance(arm, TwoLinkArm):
        theta1, theta2 = arm.inverse(x, y, elbow_up=event.button == MouseButton.LEFT)
        px, py = arm.forward(theta1, theta2)
        print(f"Target: ({x:.2f}, {y:.2f})")
        print(f"Angles: theta1 = {np.degrees(theta1):.2f}°, theta2 = {np.degrees(theta2):.2f}°")
        print(f"Error: ({abs(x - px):.2e}, {abs(y - py):.2e})")
        arm.animate(fig, (theta1, theta2), frames=100)

    elif isinstance(arm, ThreeLinkArm):
        theta1, theta2, theta3 = arm.inverse(x, y)
        px, py = arm.forward(theta1, theta2, theta3)
        print(f"Target: ({x:.2f}, {y:.2f})")
        print(f"Angles: theta1 = {np.degrees(theta1):.2f}°, theta2 = {np.degrees(theta2):.2f}°, theta3 = {np.degrees(theta3):.2f}°")
        print(f"Error: ({abs(x - px):.2e}, {abs(y - py):.2e})")
        arm.animate(fig, (theta1, theta2, theta3), frames=100)


fig.canvas.mpl_connect('button_press_event', on_click)
check.on_clicked(toggle_arm)
plot.show()