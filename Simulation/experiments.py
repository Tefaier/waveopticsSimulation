from Simulation.layer import create_line_points, uniform_layer_phases_build
from Simulation.simulation import run_layers, display_1D


def uniform_source_one_slit_1D():
    layers = []
    layers.append(create_line_points(1e-6, 1000))
    layers.append(create_line_points(1e-3, 10000))
    intensities = run_layers(uniform_layer_phases_build(layers[0], 50, 1), layers, 500e-9)
    display_1D(layers[-1], intensities)

if __name__ == '__main__':
    uniform_source_one_slit_1D()