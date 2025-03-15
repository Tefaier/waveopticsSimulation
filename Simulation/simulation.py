import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from Simulation.layer import effect_layer_by_another, calculate_mean_intensity, create_circular_appendage_filled, \
    sunflower, create_mesh_grid_slit


def run_layers(first_layer_phases: np.ndarray, layers: list[np.ndarray], wavelength: float) -> np.ndarray:
    active_layer_phases = first_layer_phases
    for i in range(1, len(layers)):
        active_layer_phases = effect_layer_by_another(layers[i], layers[i-1], active_layer_phases, wavelength)
    return calculate_mean_intensity(active_layer_phases)

def display_1D(x_axis: np.ndarray, intensities: np.ndarray):
    plt.plot(x_axis, intensities)
    plt.show()

def display_1D_extrapolate(x_axis: np.ndarray[float], intensities: np.ndarray):
    '''
    interpolation can be as follows
    'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'
    '''
    plt.imshow(intensities.reshape((1, len(intensities))).repeat(len(intensities), axis=0), cmap='viridis', interpolation='bicubic', extent=(x_axis[0], x_axis[-1], -1.0, 1.0), aspect=(x_axis[-1]))
    plt.colorbar()
    plt.show()

def display_2D(intensities: np.ndarray, extent: tuple[float, float, float, float]):
    '''
    interpolation can be as follows
    'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'
    '''
    plt.figure(figsize=(11, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(intensities, cmap='viridis', interpolation='bicubic', extent=extent)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(intensities, cmap='viridis', interpolation='bicubic', extent=extent, norm=LogNorm())
    plt.colorbar()
    plt.show()

def test():
    dots = create_mesh_grid_slit(1e-6, 5e-2, 5, 50)
    plt.figure(figsize=(5, 5))
    plt.scatter(dots[:, 0], dots[:, 1])
    plt.show()

if __name__ == "__main__":
    test()
