import math

import numpy as np

from Simulation.layer import create_line_points, uniform_layer_phases_build, create_circular_appendage_filled, \
    sunflower, point_layer_phases_build, create_mesh_grid_slit
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
    layers.append(create_line_points(1e-6, 50))
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

def uniform_source_circular_slit_2D():
    layers = []
    layers.append(create_circular_appendage_filled(0, 5e-7, 10, 300))
    screen_size = 4e-1
    screen_resolution = 50
    xv, yv, zv = np.meshgrid(np.linspace(-screen_size, screen_size, screen_resolution), np.linspace(-screen_size, screen_size, screen_resolution), 0.1)
    screen = np.concatenate(
        [
            xv.reshape((screen_resolution, screen_resolution, 1)),
            yv.reshape((screen_resolution, screen_resolution, 1)),
            zv.reshape((screen_resolution, screen_resolution, 1))
        ], axis=2).reshape((screen_resolution * screen_resolution, 3))
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 0.01), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size))

def uniform_source_circular_empty_slit_2D():
    layers = []
    shift = 0
    layers.append(create_circular_appendage_filled(4.8e-6 + shift, 5e-6 + shift, 1, 250))
    screen_size = 4e-2
    screen_resolution = 51
    xv, yv, zv = np.meshgrid(np.linspace(-screen_size, screen_size, screen_resolution), np.linspace(-screen_size, screen_size, screen_resolution), 0.1)
    screen = np.concatenate(
        [
            xv.reshape((screen_resolution, screen_resolution, 1)),
            yv.reshape((screen_resolution, screen_resolution, 1)),
            zv.reshape((screen_resolution, screen_resolution, 1))
        ], axis=2).reshape((screen_resolution * screen_resolution, 3))
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  50, 0.01), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)), (-screen_size, screen_size, -screen_size, screen_size))

def cross_2D():
    layers = []
    layers.append(np.concatenate([
        create_mesh_grid_slit(1e-4, 1e-3, 3, 100),
        create_mesh_grid_slit(1e-3, 1e-4, 100, 3)
    ], axis=0))
    screen_size = 1e-3
    screen_resolution = 101
    screen = create_mesh_grid_slit(screen_size, screen_size, screen_resolution, screen_resolution)
    screen[:, 2] = 0.1
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  50, 0.01), layers, 500e-9)
    display_2D(intensities.reshape((screen_resolution, screen_resolution)).transpose(), (-screen_size, screen_size, -screen_size, screen_size))

def triple_slit_2D():
    layers = []
    offset = 1e-6
    slit1 = create_mesh_grid_slit(3e-7, 0.1, 10, 1)
    slit2 = create_mesh_grid_slit(3e-7, 0.1, 10, 1)
    slit3 = create_mesh_grid_slit(3e-7, 0.1, 10, 1)
    slit1[:, 0] -=offset
    slit3[:, 0] += offset
    layers.append(np.concatenate([
        slit1,
        slit2,
        slit3
    ], axis=0))
    screen_size = 0.1
    screen_resolution = 1000
    screen = create_mesh_grid_slit(screen_size, 0, screen_resolution, 1)
    screen[:, 2] = 0.1
    layers.append(screen)
    intensities = run_layers(uniform_layer_phases_build(layers[0],  100, 0.01), layers, 500e-9)
    # display_1D(layers[-1][:, 0], intensities)
    display_1D_extrapolate(layers[-1][:, 0], intensities)
    # display_2D(intensities.reshape((screen_resolution, screen_resolution)).transpose(), (-screen_size, screen_size, -screen_size, screen_size))

if __name__ == '__main__':
    cross_2D()