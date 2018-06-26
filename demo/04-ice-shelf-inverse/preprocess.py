
import numpy as np
import copy

def find_points_to_fill(grid_data, mesh):
    """Find any points of a mesh where a gridded data set is missing values"""
    ny, nx = grid_data.shape

    pixel_needs_data = np.zeros((ny, nx), dtype=bool)
    coordinates = mesh.coordinates.dat.data_ro
    cells = mesh.coordinates.cell_node_map().values

    for cell in cells:
        xmin = (np.min(coordinates[cell, 0]), np.min(coordinates[cell, 1]))
        xmax = (np.max(coordinates[cell, 0]), np.max(coordinates[cell, 1]))
        imin, jmin = grid_data._index_of_point(xmin)
        imax, jmax = grid_data._index_of_point(xmax)

        for i in range(imin, imax + 2):
            for j in range(jmin, jmax + 2):
                pixel_needs_data[i, j] = grid_data.data.mask[i, j]

    nz = pixel_needs_data.nonzero()
    return list(zip(nz[0], nz[1]))


def fill_missing_points(grid_data, indices, radius):
    p = copy.deepcopy(grid_data)

    ny, nx = grid_data.shape
    for i, j in indices:
        total_weight = 0.0
        p.data[i, j] = 0.0
        p.data.mask[i, j] = False

        for k in range(max(i - radius, 0), min(i + radius + 1, ny)):
            for l in range(max(j - radius, 0), min(j + radius + 1, nx)):
                if not grid_data.data.mask[k, l]:
                    weight = max(1 - abs(k - i)/radius, 1 - abs(l - j)/radius)
                    total_weight += weight
                    p.data[i, j] += weight * grid_data.data[k, l]

        if total_weight == 0.0:
            raise ValueError("No close data to fill {0}, {1}".format(i, j))
        p.data[i, j] /= total_weight

    return p


def preprocess(grid_data, mesh, radius=4):
    indices = find_points_to_fill(grid_data, mesh)
    return fill_missing_points(grid_data, indices, radius)

