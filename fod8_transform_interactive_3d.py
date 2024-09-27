import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

COLORS = ['r', 'g', 'b', 'y', 'c']  # Colors for each point
Z = 1

# Original points of the square with z=1
original_points = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]])

# Identity matrix for initial transformation
transform_matrix_i = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def cart_affine_point_transformation(t_mat: np.array, p: np.array) -> np.array:
    '''returns transformed point in cartesian form: (x,y,z)'''
    return np.dot(t_mat, p)

# Update function for updating the transformation and redrawing
def update(val) -> None:
    # Create new transformation matrix
    new_transform_matrix = np.array([
        [a_slider.val, b_slider.val, c_slider.val],
        [d_slider.val, e_slider.val, f_slider.val],
        [g_slider.val, h_slider.val, 1]
    ])

    axe.clear()
    axe.set_box_aspect([1,1,1])
    axe.grid(True)
    axe.set_xlim(-2, 3)
    axe.set_ylim(-2, 3)
    axe.set_zlim(-2, 3)

    points = original_points
    transformed_points = np.array([cart_affine_point_transformation(new_transform_matrix, p) for p in points])

    # Create scatter plot for transformed points
    axe.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], color='b')

    # Plot original points as red 'x'
    axe.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', marker='x', label='Original Points')

    # Lines connecting original and transformed points
    for i in range(len(points)):
        axe.plot([points[i, 0], transformed_points[i, 0]], 
                 [points[i, 1], transformed_points[i, 1]], 
                 [points[i, 2], transformed_points[i, 2]], 'k--')

    # Create polygons for original and transformed shapes
    original_polygon = Poly3DCollection([points[:, :3]], color='r', alpha=0.5)
    transformed_polygon = Poly3DCollection([transformed_points[:, :3]], color='b', alpha=0.5)
    axe.add_collection3d(original_polygon)
    axe.add_collection3d(transformed_polygon)
    plt.draw()

# Create the figure and axes
fig = plt.figure()
axe = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.35)

# Create sliders for each transformation parameter
axcolor = 'lightgoldenrodyellow'
a_ax = plt.axes([0.1, 0.25, 0.2, 0.03], facecolor=axcolor)
b_ax = plt.axes([0.4, 0.25, 0.2, 0.03], facecolor=axcolor)
c_ax = plt.axes([0.7, 0.25, 0.2, 0.03], facecolor=axcolor)
d_ax = plt.axes([0.1, 0.15, 0.2, 0.03], facecolor=axcolor)
e_ax = plt.axes([0.4, 0.15, 0.2, 0.03], facecolor=axcolor)
f_ax = plt.axes([0.7, 0.15, 0.2, 0.03], facecolor=axcolor)
g_ax = plt.axes([0.1, 0.05, 0.2, 0.03], facecolor=axcolor)
h_ax = plt.axes([0.4, 0.05, 0.2, 0.03], facecolor=axcolor)

a_slider = Slider(a_ax, 'a', -10.0, 10.0, valinit=1)
b_slider = Slider(b_ax, 'b', -10.0, 10.0, valinit=0)
c_slider = Slider(c_ax, 'c', -10.0, 10.0, valinit=0)
d_slider = Slider(d_ax, 'd', -10.0, 10.0, valinit=0)
e_slider = Slider(e_ax, 'e', -10.0, 10.0, valinit=1)
f_slider = Slider(f_ax, 'f', -10.0, 10.0, valinit=0)
g_slider = Slider(g_ax, 'g', -10.0, 10.0, valinit=0)
h_slider = Slider(h_ax, 'h', -10.0, 10.0, valinit=0)

# Attach the update function to slider changes
a_slider.on_changed(update)
b_slider.on_changed(update)
c_slider.on_changed(update)
d_slider.on_changed(update)
e_slider.on_changed(update)
f_slider.on_changed(update)
g_slider.on_changed(update)
h_slider.on_changed(update)

# Initial plot
update(None)

plt.show()