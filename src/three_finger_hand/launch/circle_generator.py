import numpy as np
import matplotlib.pyplot as plt

def generate_circle_points(center, normal, radius, N):
    center = np.array(center)
    normal = np.array(normal) / np.linalg.norm(normal)

    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)

    thetas = np.linspace(0, 2 * np.pi, N)
    points = []
    for theta in thetas:
        p = center + radius * np.cos(theta) * u + radius * np.sin(theta) * v
        points.append(p)

    return np.array(points)

def plot_circle(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Circle Path')
    plt.show()

# Sampling parameters:
center_point = [0.06, 0.0, 0.05]
normal_vec = [0, 0.6, 0.8]
r = 0.03
n_points = 50

path = generate_circle_points(center_point, normal_vec, r, n_points)
print(f"{len(path)} waypoints generated.")
plot_circle(path)
