import math

import numpy as np

from Simulation.layer import create_line_points, uniform_layer_phases_build, create_circular_appendage_filled, \
    sunflower, point_layer_phases_build, create_mesh_grid_slit
from Simulation.simulation import run_layers, display_1D, display_1D_extrapolate, display_2D


def uniform_source_one_slit_1D():
    layers = []
    layers.append(create_line_points(2e-6, 500))
    screen = create_line_points(0.14, 500)
    screen[:, 2] = 0.05
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 0.001), layers, 700e-9)
    display_1D(layers[-1][:, 0], intensities)

def uniform_source_one_slit_1D_extrapolated():
    layers = []
    layers.append(create_line_points(2e-6, 500))
    screen = create_line_points(0.14, 500)
    screen[:, 2] = 0.05
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 0.001), layers, 500e-9)
    display_1D_extrapolate(layers[-1][:, 0], intensities, True)

def uniform_source_one_slit_2D():
    layers = []
    layers.append(create_mesh_grid_slit(2e-6, 0, 100, 1))
    screen_size = 0.14
    screen_resolution = 101
    screen = create_mesh_grid_slit(screen_size, screen_size, screen_resolution, screen_resolution)
    screen[:, 2] = 0.05
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 0.001), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size), True)

def uniform_source_circular_slit_2D():
    layers = []
    layers.append(sunflower(0, 90e-6, 500))
    screen_size = 0.002
    screen_resolution = 201
    screen = create_mesh_grid_slit(screen_size, screen_size, screen_resolution, screen_resolution)
    screen[:, 2] = 0.05
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 0.01), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size), True)

def uniform_source_circular_empty_slit_2D():
    layers = []
    shift = 1e-4
    layers.append(create_circular_appendage_filled(shift, 90e-6 + shift, 1, 200))
    screen_size = 0.03
    screen_resolution = 51
    screen = create_mesh_grid_slit(screen_size, screen_size, screen_resolution, screen_resolution)
    screen[:, 2] = 1
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  50, 0.01), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size), True)

def cross_2D():
    layers = []
    layers.append(np.concatenate([
        create_mesh_grid_slit(100e-6, 100e-5, 10, 100),
        create_mesh_grid_slit(100e-5, 100e-6, 100, 10)
    ], axis=0))
    screen_size = 0.005
    screen_resolution = 201
    screen = create_mesh_grid_slit(screen_size, screen_size, screen_resolution, screen_resolution)
    screen[:, 2] = 0.3
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  50, 0.001), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size), True)

def triple_slit_2D():
    layers = []
    offset = 5e-6
    slit1 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit2 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit3 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit1[:, 0] -=offset
    slit3[:, 0] += offset
    layers.append(np.concatenate([
        slit1,
        slit2,
        slit3
    ], axis=0))
    screen_size = 0.1
    screen_resolution = 500
    screen = create_mesh_grid_slit(screen_size, 0, screen_resolution, 1)
    screen[:, 2] = 0.05
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  100, 0.01), layers, 500e-9)
    # display_1D(layers[-1][:, 0], intensities)
    display_1D_extrapolate(layers[-1][:, 0], intensities)
    # display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size))

def five_slit_2D():
    layers = []
    offset = 4e-6
    slit1 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit2 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit3 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit4 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit5 = create_mesh_grid_slit(2e-6, 0, 30, 1)
    slit1[:, 0] -= 2*offset
    slit2[:, 0] -= offset
    slit4[:, 0] += offset
    slit5[:, 0] += 2*offset
    layers.append(np.concatenate([
        slit1,
        slit2,
        slit3,
        slit4,
        slit5
    ], axis=0))
    screen_size = 0.1
    screen_resolution = 1000
    screen = create_mesh_grid_slit(screen_size, 0, screen_resolution, 1)
    screen[:, 2] = 0.05
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  100, 0.01), layers, 500e-9)
    display_1D(layers[-1][:, 0], intensities)
    # display_1D_extrapolate(layers[-1][:, 0], intensities)
    # display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size))

def shifting_source_circular_empty_slit_2D():
    layers = []
    shift = 5e-5
    layers.append(np.array([[35e-5, 35e-5, -0.1]]))
    layers.append(create_circular_appendage_filled(shift, 90e-6 + shift, 1, 1000))
    screen_size = 0.05
    screen_resolution = 51
    screen = create_mesh_grid_slit(screen_size, screen_size, screen_resolution, screen_resolution)
    screen[:, 2] = 1
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  50, 0.01), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size), True)

def two_layer_slit_2D():
    layers = []
    offset = 30e-6
    slit1 = create_mesh_grid_slit(10e-6, 0, 100, 1)
    slit2 = create_mesh_grid_slit(10e-6, 0, 100, 1)
    slit1[:, 0] -=offset
    slit2[:, 0] += offset
    layer = np.concatenate([
        slit1,
        slit2
    ], axis=0)
    layer[:, 2] -= 0.1
    layers.append(layer)
    slit1 = create_mesh_grid_slit(10e-6, 0, 300, 1)
    slit2 = create_mesh_grid_slit(10e-6, 0, 300, 1)
    offset = 30e-6
    slit1[:, 0] -= offset
    slit2[:, 0] += offset
    slit1[:, 0] += 100e-7
    slit2[:, 0] += 100e-7
    layers.append(np.concatenate([
        slit1,
        slit2
    ], axis=0))
    screen_size = 0.1
    screen_resolution = 1000
    screen = create_mesh_grid_slit(screen_size, 0, screen_resolution, 1)
    screen[:, 2] = 0.5
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  100, 0.00001), layers, 500e-9)
    # display_1D(layers[-1][:, 0], intensities)
    display_1D_extrapolate(layers[-1][:, 0], intensities)
    # display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size))

def uniform_source_square_2D():
    layers = []
    layers.append(create_mesh_grid_slit(1e-5, 1e-5, 20, 20))
    screen_size = 0.14
    screen_resolution = 101
    screen = create_mesh_grid_slit(screen_size, screen_size, screen_resolution, screen_resolution)
    screen[:, 2] = 0.05
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 0.001), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size), True)

if __name__ == '__main__':
    uniform_source_square_2D()