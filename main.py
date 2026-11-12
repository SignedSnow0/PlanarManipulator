import matplotlib.pyplot as plot
import matplotlib.patches as patches
import matplotlib.backend_bases as MouseButton
from matplotlib.animation import FuncAnimation
from shapely.geometry import MultiPolygon, Point
from shapely.ops import polygonize, unary_union 
import numpy as np

class Arm:
    def __init__(self, positon, lenght, current_angle, angle_min, angle_max, attachment):
        self.position = positon
        self.length = lenght
        self.angle = current_angle
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.attachment = None

        if attachment is not None:
            self.attachment = attachment

    def draw(self, ax):
        if self.attachment is not None:
            self.position = self.attachment.get_end_position()
        x = self.position[0] + self.length * np.cos(self.angle)
        y = self.position[1] + self.length * np.sin(self.angle)
        self.line = ax.plot([self.position[0], x], [self.position[1], y], 'o-', lw=2)

    def get_end_position(self):
        x = self.position[0] + self.length * np.cos(self.angle)
        y = self.position[1] + self.length * np.sin(self.angle)
        return (x, y)

    def move(self, new_angle):
        self.angle = new_angle
        x = self.position[0] + self.length * np.cos(self.angle)
        y = self.position[1] + self.length * np.sin(self.angle)
        if self.attachment is None:
            self.line[0].set_xdata([self.position[0], x])
            self.line[0].set_ydata([self.position[1], y])
        else:
            self.position = self.attachment.get_end_position()
            self.line[0].set_xdata([self.position[0], x])
            self.line[0].set_ydata([self.position[1], y])

def update(frame):
    arm1.move(angles1[frame])
    arm2.move(angles2[frame])

    return arm1.line + arm2.line

def calculate_workspace(arm):
    max_radius = 0
    max_length = 0
    current_arm = arm
    while current_arm.attachment is not None:
        max_radius += current_arm.length
        if current_arm.length > max_length:
            max_length = current_arm.length
        current_arm = current_arm.attachment

    max_radius += current_arm.length
    if current_arm.length > max_length:
        max_length = current_arm.length

    min_radius = max(0, max_length - (max_radius - max_length))

    return min_radius, max_radius

def calculate_workspace_2_link(arm):
    arm1 = arm.attachment
    arm2 = arm

    l1 = arm1.length
    l2 = arm2.length

    r_arc1 = np.sqrt(l1**2 + l2**2 + 2 * l1 * l2 * np.cos(arm2.angle_min))
    r_arc2 = np.sqrt(l1**2 + l2**2 + 2 * l1 * l2 * np.cos(arm2.angle_max))

    circle_1 = (0, 0, r_arc1)
    circle_2 = (0, 0, r_arc2)
    circle_3 = (l1* np.cos(arm1.angle_min), l1 * np.sin(arm1.angle_min), l2)
    circle_4 = (l1* np.cos(arm1.angle_max), l1 * np.sin(arm1.angle_max), l2)

    return [circle_1, circle_2, circle_3, circle_4]


def is_point_reachable(x, y, arm, tol=1e-7):
    arm1 = arm.attachment
    arm2 = arm
    l1, l2 = arm1.length, arm2.length

    r2 = x**2 + y**2
    cos_t2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)

    if abs(cos_t2) > 1.0 + tol:
        return False
    cos_t2 = np.clip(cos_t2, -1.0, 1.0)

    for sign in [1.0, -1.0]:
        t2 = sign * np.arccos(cos_t2)
        if not (arm2.angle_min - tol <= t2 <= arm2.angle_max + tol):
            continue

        t1 = np.arctan2(y, x) - np.arctan2(
            l2 * np.sin(t2), l1 + l2 * np.cos(t2)
        )
        while t1 < arm1.angle_min - np.pi:
            t1 += 2 * np.pi
        while t1 > arm1.angle_max + np.pi:
            t1 -= 2 * np.pi

        if arm1.angle_min - tol <= t1 <= arm1.angle_max + tol:
            return True
    return False

def build_workspace_shapely(arm, resolution=128):
    circles = calculate_workspace_2_link(arm)
    
    arm1, arm2 = arm.attachment, arm
    l1, l2 = arm1.length, arm2.length

    if arm2.angle_min < 0 < arm2.angle_max:
        circles.append((0, 0, l1 + l2))
    if (arm2.angle_min < np.pi < arm2.angle_max) or (
        arm2.angle_min < -np.pi < arm2.angle_max
    ):
        circles.append((0, 0, abs(l1 - l2)))

    rings = [
        Point(cx, cy).buffer(r, resolution=resolution).exterior
        for cx, cy, r in circles
        if r > 0
    ]
    noded_lines = unary_union(rings)

    valid_faces = [
        face
        for face in polygonize(noded_lines)
        if is_point_reachable(
            face.representative_point().x, face.representative_point().y, arm
        )
    ]

    return unary_union(valid_faces), circles

def inverse_2_link(arm, target, elbow_up=True):
    l1 = arm.attachment.length
    l2 = arm.length
    x, y = target

    theta2 = np.arccos((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
    if elbow_up:
        theta2 = -theta2

    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    return theta1, theta1 + theta2

def forward_2_link(l1, l2, theta1, theta2):
    x = l1 * np.cos(theta1) + l2 * np.cos(theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta2)

    return (x, y)

def animate_arm(from_angles, to_angles, frames=100):
    global angles1, angles2
    angles1 = np.linspace(from_angles[0], to_angles[0], frames)
    angles2 = np.linspace(from_angles[1], to_angles[1], frames)
    FuncAnimation(fig, update, frames=frames, interval=20, blit=True, repeat=False)
    plot.show()

def on_click(event):
    if event.button == MouseButton.MouseButton.LEFT or event.button == MouseButton.MouseButton.RIGHT:
        if event.xdata is not None and event.ydata is not None:
            if not is_point_reachable(event.xdata, event.ydata, arm2):
                print(f"Target: ({event.xdata:.2f}, {event.ydata:.2f}) is not reachable.")
                return

            (theta1, theta2) = inverse_2_link(arm2, (event.xdata, event.ydata), elbow_up=event.button == MouseButton.MouseButton.LEFT)
            (x, y) = forward_2_link(arm1.length, arm2.length, theta1, theta2)
            ex = abs(event.xdata - x)
            ey = abs(event.ydata - y)
            print(f"Target: ({event.xdata:.2f}, {event.ydata:.2f})")
            print(f"Calculated angles: theta1 = {np.degrees(theta1):.2f} degrees, theta2 = {np.degrees(theta2):.2f} degrees")
            print(f"Error: ({ex}, {ey})")

            animate_arm((arm1.angle, arm2.angle), (theta1, theta2), frames=100)

def setup_workspace(ax, arm):
    (min_radius, max_radius) = calculate_workspace(arm)
    ax.set_facecolor('lightgray')
    min_circle = patches.Circle((0, 0), min_radius, color='lightgray')
    max_circle = patches.Circle((0, 0), max_radius, color='white')
    ax.add_patch(max_circle)
    ax.add_patch(min_circle)
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()

def plot_workspace(arm, ax):
    workspace, circles = build_workspace_shapely(arm)

    polys = (
        workspace.geoms
        if isinstance(workspace, MultiPolygon)
        else [workspace]
    )

    ax.set_facecolor('lightgray')
    for poly in polys:
        x_ext, y_ext = poly.exterior.xy
        ax.fill(x_ext, y_ext, color='white', label='Workspace')
        ax.plot(x_ext, y_ext, color='white')

        for interior in poly.interiors:
            x_int, y_int = interior.xy
            ax.fill(x_int, y_int, color='lightgray')
            ax.plot(x_int, y_int, color='lightgray')

    max_reach = arm.attachment.length + arm.length
    ax.set_xlim(-max_reach * 1.15, max_reach * 1.15)
    ax.set_ylim(-max_reach * 1.15, max_reach * 1.15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

angles1 = []
angles2 = []

arm1 = Arm((0, 0), 7, np.pi / 4, -np.pi / 2, np.pi, None)
arm2 = Arm((0, 0), 3, np.pi / 3, -np.pi, np.pi / 2, arm1)
fig, ax = plot.subplots(figsize=(6, 6))

plot_workspace(arm2, ax)

arm1.draw(ax)
arm2.draw(ax)

plot.connect('button_press_event', on_click)
plot.show()