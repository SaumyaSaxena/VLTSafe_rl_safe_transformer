import numpy as np

def signed_dist_fn_rectangle(grid_x, x_target_min, x_target_max, obstacle=False, plot=False):
    # Compute distances to each edge of the rectangle
    dist_from_walls = np.maximum(x_target_min - grid_x, grid_x - x_target_max)
    signed_distance_grid = np.max(dist_from_walls, axis=-1)
    if obstacle:
        signed_distance_grid = -1*signed_distance_grid
    return signed_distance_grid

def signed_dist_fn_rectangle_obstacle(grid_x, x_target_min, x_target_max, obstacle=True):
    # Compute distances to each edge of the rectangle
    # g(x)>0 is obstacle
    dist_from_walls = np.minimum(grid_x - x_target_min, x_target_max - grid_x)
    signed_distance_grid = np.min(dist_from_walls, axis=-1)
    return signed_distance_grid

def create_grid(x_min, x_max, N_x):
    X = [np.linspace(x_min[i], x_max[i], N_x[i]) for i in range(len(x_min))]
    grid = np.meshgrid(*X, indexing='ij')
    return np.stack(grid, axis=-1)

def create_centered_polygon_with_halfsize(size_x: float, size_y: float):
    return np.array([
        [-size_x, -size_y],
        [-size_x, size_y],
        [size_x, size_y],
        [size_x, -size_y],
    ])

# Compute bounding boxes for all objects
def get_bounding_boxes(segmentation):
    unique_objects = np.unique(segmentation)
    bboxes = {}

    for obj_id in unique_objects:
        if obj_id == 0:
            continue  # Ignore background
        ys, xs = np.where(segmentation == obj_id)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bboxes[obj_id] = (x_min, y_min, x_max, y_max)

    return bboxes

