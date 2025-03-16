import math

import numpy as np

# https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(x, y):  # makes array with (y_size, x_size, 2, 3)
    dim_x = len(x)
    dim_y = len(y)
    dim_info = x.shape[-1]
    x_r = np.tile(x, (dim_y, 1)).reshape((dim_y, dim_x, dim_info))
    y_r = np.repeat(y, dim_x, axis=0).reshape((dim_y, dim_x, dim_info))
    return np.concatenate([x_r, y_r], axis=2).reshape((dim_y, dim_x, 2, dim_info))

def point_layer_phases_build(phases_count: int, amplitude: float) -> np.ndarray[float]:
    info = np.zeros((1, phases_count))
    info[0, 0] = amplitude
    return info

def uniform_layer_phases_build(layer: np.ndarray, phases_count: int, amplitude: float) -> np.ndarray[float]:
    return np.tile(point_layer_phases_build(phases_count, amplitude), (layer.shape[0], 1))

# 40 MB +-
maximum_calculate_size = 5e6

def effect_layer_by_another(
        to_effect: np.ndarray,  # must be [[x, y, z]]
        effect_by_pos: np.ndarray,  # must be [[x, y, z]]
        effect_by_phases: np.ndarray,  # must be [[amplt at offset 0, ..., amplt at offset 2pi]]
        wavelength: float
) -> np.ndarray:  # will be [[amplt at offset 0, ..., amplt at offset 2pi]] for to_effect
    to_effect_parts = np.array_split(to_effect, np.ceil(to_effect.shape[0] * effect_by_phases.size / maximum_calculate_size), axis=0)
    to_effect_phases = []
    for part in to_effect_parts:
        phase_resolution =  effect_by_phases.shape[1]
        cartesian_phases = np.tile(effect_by_phases, (len(part), 1)).reshape((len(effect_by_pos), len(part), phase_resolution))

        cartesian_positions = cartesian_product(part, effect_by_pos)
        distances = (cartesian_positions[:, :, 0, :] - cartesian_positions[:, :, 1, :])
        distances = np.sqrt((distances*distances).sum(axis=2))
        # cartesian_phases *= np.exp(distances * -1)[:, :, np.newaxis]
        # cartesian_phases /= distances[:, :, np.newaxis]  # TODO this method has a problem related to amplitude skyrocketing if distance is small while it must not theoretically exceed source amplitude
        cartesian_phases /= (distances + 1)[:, :, np.newaxis]

        offsets = np.mod(np.floor(phase_resolution * distances / wavelength), phase_resolution).astype(int)
        rows, columns, thr_indices = np.ogrid[:cartesian_phases.shape[0], :cartesian_phases.shape[1], :cartesian_phases.shape[2]]
        thr_indices = thr_indices - offsets[:, :, np.newaxis]
        cartesian_phases = cartesian_phases[rows, columns, thr_indices]
        to_effect_phases.append(np.sum(cartesian_phases, axis=0))
    return np.concatenate(to_effect_phases, axis=0)


# https://mathworld.wolfram.com/HarmonicAdditionTheorem.html
# input as [[A1, A2, ..., An]]
def calculate_mean_intensity(layer_phases: np.ndarray) -> np.ndarray[float]:
    layer_parts = np.array_split(layer_phases, np.ceil(layer_phases.size * layer_phases.shape[1] / maximum_calculate_size), axis=0)
    result_parts = []
    for part in layer_parts:
        phase_resolution =  part.shape[1]
        phase_offsets = np.linspace(0, np.pi * 2, phase_resolution)
        cartesian_phases = np.transpose([np.tile(phase_offsets, phase_resolution), np.repeat(phase_offsets, phase_resolution)])
        cos_values = np.cos(cartesian_phases[:, 0] - cartesian_phases[:, 1])
        result_parts.append((np.repeat(part, phase_resolution, axis=1) * np.tile(part, phase_resolution) * cos_values[np.newaxis, :]).sum(axis=1))
    return np.concatenate(result_parts, axis=0)

def create_line_points(width: float, resolution: int) -> np.ndarray[float]:
    dots = np.zeros((resolution, 3))
    dots[:, 0] = np.linspace(-width/2, width/2, resolution)
    return dots

# https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(x, y):
    rho = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)
    return (rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.concatenate([x.reshape((1, len(x))), y.reshape((1, len(x)))], axis=0).transpose()

# by default creates [[x, y, 0]] with center at [0, 0, 0]
def create_circular_appendage(radius: float, resolution: int):
    return np.concatenate([pol2cart(np.ones((resolution, ), dtype=float) * radius, np.linspace(0, 2 * np.pi * (resolution - 1) / resolution, resolution)), np.zeros((resolution, 1))], axis=1)

def create_circular_appendage_filled(radius_from: float, radius_to: float, radius_resolution: int, resolution_total: int):
    radiuses = np.linspace(radius_from, radius_to, radius_resolution)
    numbers_at_radiuses = radiuses.copy()
    numbers_at_radiuses *= (resolution_total / numbers_at_radiuses.sum())
    if (radius_from == 0):
        numbers_at_radiuses[0] = 1
    numbers_at_radiuses = np.floor(numbers_at_radiuses).astype(int)
    numbers_at_radiuses[-1] += (resolution_total - numbers_at_radiuses.sum())
    return np.concatenate([create_circular_appendage(radiuses[index], numbers_at_radiuses[index]) for index in range(0, radius_resolution)], axis = 0)

def create_mesh_grid_slit(size_x: float, size_y: float, resolution_x: int, resolution_y: int):
    xv, yv, zv = np.meshgrid(np.linspace(-size_x/2, size_x/2, resolution_x), np.linspace(-size_y/2, size_y/2, resolution_y), 0)
    dots = np.concatenate(
        [
            xv.reshape((resolution_x, resolution_y, 1)),
            yv.reshape((resolution_x, resolution_y, 1)),
            zv.reshape((resolution_x, resolution_y, 1))
        ], axis=2).reshape((resolution_x * resolution_y, 3))
    return dots

phi = (1 + math.sqrt(5)) / 2  # golden ratio

def sunflower(radius_from: float, radius_to: float, resolution: int, alpha=0):
    points = []
    angle_stride = 2 * math.pi / phi ** 2
    sunflower_to = int(resolution * (radius_to * radius_to) / (radius_to * radius_to - radius_from * radius_from))
    b = round(alpha * math.sqrt(resolution))  # number of boundary points
    for k in range(sunflower_to - resolution + 1, sunflower_to + 1):
        r = radius(k, sunflower_to, b)
        theta = k * angle_stride
        points.append((r * math.cos(theta), r * math.sin(theta)))
    points = np.array(points)
    points *= radius_to
    return np.concatenate([points, np.zeros((len(points), 1))], axis = 1)

def radius(k, n, b):
    if k > n - b:
        return 1.0
    else:
        return math.sqrt(k - 0.5) / math.sqrt(n - (b + 1) / 2)