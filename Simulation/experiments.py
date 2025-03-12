import numpy as np

from Simulation.layer import create_line_points, uniform_layer_phases_build
from Simulation.simulation import run_layers, display_1D, display_1D_extrapolate, display_2D


def uniform_source_one_slit_1D():
    layers = []
    layers.append(create_line_points(1e-6, 100))
    screen = create_line_points(5e-1, 100)
    screen[:, 2] = 0.1
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 1), layers, 500e-9)
    display_1D(layers[-1][:, 0], intensities)

def uniform_source_one_slit_1D_extrapolated():
    layers = []
    layers.append(create_line_points(1e-6, 100))
    screen = create_line_points(5e-1, 100)
    screen[:, 2] = 0.1
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 1), layers, 500e-9)
    display_1D_extrapolate(layers[-1][:, 0], intensities)

def uniform_source_one_slit_2D():
    layers = []
    layers.append(create_line_points(1e-6, 5))
    screen_size = 5e-1
    screen_resolution = 50
    xv, yv, zv = np.meshgrid(np.linspace(-screen_size, screen_size, screen_resolution), np.linspace(-screen_size, screen_size, screen_resolution), 0.1)
    screen = np.concatenate(
        [
            xv.reshape((screen_resolution, screen_resolution, 1)),
            yv.reshape((screen_resolution, screen_resolution, 1)),
            zv.reshape((screen_resolution, screen_resolution, 1))
        ], axis=2).reshape((screen_resolution * screen_resolution, 3))
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 1), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size))

if __name__ == '__main__':
    uniform_source_one_slit_2D()