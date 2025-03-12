import matplotlib.pyplot as plt
import numpy as np

from Simulation.layer import effect_layer_by_another, calculate_mean_intensity

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
    plt.imshow(intensities.repeat(len(intensities), axis=0).reshape((len(intensities), len(intensities))), cmap='viridis', interpolation='nearest', extent=(x_axis[0], x_axis[-1], -1.0, 1.0))
    plt.colorbar()
    plt.show()
