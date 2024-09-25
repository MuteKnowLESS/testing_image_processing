import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

COLORS = ['r', 'g', 'b', 'y', 'c']  # Colors for each point
Z = 1

# Original points of the square
original_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

# Identity matrix for initial transformation
transform_matrix_i = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def cart_affine_point_transformation(t_mat: np.array, p: np.array) -> np.array:
    '''returns transformed point in cartesian form: (x,y)'''
    transformed_point = np.dot(t_mat, p)
    return transformed_point[:2] / transformed_point[2]

def homo_affine_point_transformation(t_mat: np.array, p: np.array) -> np.array:
    '''returns transformed point in homogenous form: (x,y,z)'''
    return np.dot(t_mat, p)

# Update function for updating the transformation and redrawing
def update(val) -> None:
    # Create new transformation matrix
    new_transform_matrix = np.array([
        [a_slider.val, b_slider.val, c_slider.val],
        [d_slider.val, e_slider.val, f_slider.val],
        [g_slider.val, h_slider.val, 1]
    ])

    # new_transform_matrix = np.array([
    #     [a_slider.val, b_slider.val, 0],
    #     [d_slider.val, e_slider.val, 0],
    #     [g_slider.val, h_slider.val, 1]
    # ])

    axe.clear()
    axe.axis('equal')
    axe.grid(True)
    axe.set_ylim(-3, 4)
    axe.set_xlim(-3, 4)

    points = original_points
    transformed_points = np.array([cart_affine_point_transformation(new_transform_matrix, np.append(p, Z)) for p in points])

    # Create scatter plot for transformed points
    axe.scatter(transformed_points[:, 0], transformed_points[:, 1], color='b')

    # Plot original points as red 'x'
    axe.plot(points[:, 0], points[:, 1], 'rx', label='Original Points')

    # Lines connecting original and transformed points
    [axe.plot([points[i, 0], transformed_points[i, 0]], 
              [points[i, 1], transformed_points[i, 1]], 'k--')[0] for i in range(len(points))]

    # Create polygons for original and transformed shapes
    original_polygon = plt.Polygon(points, closed=True, fill=None, edgecolor='r')
    transformed_polygon = plt.Polygon(transformed_points, closed=True, fill=None, edgecolor='b')
    axe.add_patch(original_polygon)
    axe.add_patch(transformed_polygon)
    plt.draw()

# Create the figure and axes
fig, axe = plt.subplots()
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
