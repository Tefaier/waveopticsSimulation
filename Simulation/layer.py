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

def effect_layer_by_another(
        to_effect: np.ndarray,  # must be [[x, y, z]]
        effect_by_pos: np.ndarray,  # must be [[x, y, z]]
        effect_by_phases: np.ndarray,  # must be [[amplt at offset 0, ..., amplt at offset 2pi]]
        wavelength: float
) -> np.ndarray:  # will be [[amplt at offset 0, ..., amplt at offset 2pi]] for to_effect
    phase_resolution =  effect_by_phases.shape[1]
    cartesian_phases = np.tile(effect_by_phases, (len(to_effect), 1)).reshape((len(effect_by_pos), len(to_effect), phase_resolution))

    cartesian_positions = cartesian_product(to_effect, effect_by_pos)
    distances = cartesian_positions[:, :, 0, :] - cartesian_positions[:, :, 1, :]
    distances = np.sqrt((distances*distances).sum(axis=2))
    cartesian_phases *= np.exp(distances * -1)[:, :, np.newaxis]
    #cartesian_phases /= distances[:, :, np.newaxis]  # TODO this method has a problem related to amplitude skyrocketing if distance is small while it must not theoretically exceed source amplitude

    offsets = np.mod(np.floor(phase_resolution * distances / wavelength), phase_resolution).astype(int)
    rows, columns, thr_indices = np.ogrid[:cartesian_phases.shape[0], :cartesian_phases.shape[1], :cartesian_phases.shape[2]]
    thr_indices = thr_indices - offsets[:, :, np.newaxis]
    cartesian_phases = cartesian_phases[rows, columns, thr_indices]
    return np.sum(cartesian_phases, axis=0)


# https://mathworld.wolfram.com/HarmonicAdditionTheorem.html
# input as [[A1, A2, ..., An]]
def calculate_mean_intensity(layer_phases: np.ndarray) -> np.ndarray[float]:
    phase_resolution =  layer_phases.shape[1]
    phase_offsets = np.linspace(0, np.pi * 2, phase_resolution)
    cartesian_phases = np.transpose([np.tile(phase_offsets, phase_resolution), np.repeat(phase_offsets, phase_resolution)])
    cos_values = np.cos(cartesian_phases[:, 0] - cartesian_phases[:, 1])
    return (np.repeat(layer_phases, phase_resolution, axis=1) * np.tile(layer_phases, phase_resolution) * cos_values[np.newaxis, :]).sum(axis=1)

def create_line_points(width: float, resolution: int) -> np.ndarray[float]:
    dots = np.zeros((resolution, 3))
    dots[:, 0] = np.linspace(-width/2, width/2, resolution)
    return dots
