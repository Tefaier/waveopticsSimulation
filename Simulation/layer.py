import numpy as np

# https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(x, y):  # makes array with (y_size, x_size, 2, 3)
    dim_x = len(x)
    dim_y = len(y)
    x_r = np.tile(x, (dim_y, 1)).reshape((dim_y, dim_x, 3))
    y_r = np.repeat(y, dim_x, axis=1).reshape((dim_y, dim_x, 3))
    return np.concatenate([x_r, y_r], axis=2).reshape((dim_y, dim_x, 2, 3))

def point_layer_build(phases_count: int, amplitude: float) -> np.ndarray:
    info = np.zeros((1, phases_count))
    info[0, 0] = amplitude
    return info

def uniform_layer_build(layer: np.ndarray, phases_count: int, amplitude: float) -> np.ndarray:
    return np.tile(point_layer_build(phases_count, amplitude), (layer.shape[0], 1))

def effect_layer_by_another(
        to_effect: np.ndarray,  # must be [[x, y, z]]
        effect_by_pos: np.ndarray,  # must be [[x, y, z]]
        effect_by_phases: np.ndarray,  # must be [[amplt at offset 0, ..., amplt at offset 2pi]]
        wavelength: float
) -> np.ndarray:  # will be [[amplt at offset 0, ..., amplt at offset 2pi]] for to_effect
    phase_resolution =  effect_by_phases.shape[1]
    cartesian_positions = cartesian_product(to_effect, effect_by_pos)
    distances = cartesian_positions[:, :, 0, :] - cartesian_positions[:, :, 1, :]
    distances = (distances*distances).sum(axis=2)**0.5
    offsets = np.floor(np.mod(distances / wavelength, 1) * phase_resolution)
    cartesian_phases = np.repeat(effect_by_phases, len(to_effect), axis=1).reshape((len(effect_by_pos), len(to_effect), phase_resolution))
    cartesian_phases /= distances[:, None]  # TODO this method has a problem related to amplitude skyrocketing if distance is small while it must not theoretically exceed source amplitude
    rows, columns, thr_indices = np.ogrid[:cartesian_phases.shape[0], :cartesian_phases.shape[1], :cartesian_phases.shape[2]]
    thr_indices = thr_indices - offsets[:, :, np.newaxis]
    cartesian_phases = cartesian_phases[rows, columns, thr_indices]
    cartesian_phases = cartesian_phases.transpose(1, 0, 2)
    cartesian_phases = np.sum(cartesian_phases, axis=1)
    return cartesian_phases