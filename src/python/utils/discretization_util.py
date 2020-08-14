import copy

import numpy as np

"""Collection of functions to disretize a neural network"""


def get_naive_discretized_network(network):
    """Naive discretization.

    Discretize the input network by finding the common exponential across a layer and
    multiplying all the values with 10^exponent.

    Parameters
    ----------
    network : MultiClassNeuralNetwork
        the neural network to be discretized
    Returns
    -------
    out : MultiClassNeuralNetwork
        the discretized model
    """
    discretized_network = copy.deepcopy(network)
    for layer in discretized_network.layers:
        weights_scale_factor = np.abs(np.round(np.mean(np.log10(np.abs(layer.weights) + 1e-10))))
        biases_scale_factor = np.abs(np.round(np.mean(np.log10(np.abs(layer.biases) + 1e-10))))
        layer.weights = np.round(layer.weights * 10 ** weights_scale_factor)
        layer.biases = np.round(layer.biases * 10 ** biases_scale_factor)
        layer.weights = layer.weights.astype(np.int32)
        layer.biases = layer.biases.astype(np.int32)
    return discretized_network


def get_quantized_network(network, power=7, array_type=np.int32):
    """Quantization.

    Quantize the input network by scaling the network weights and biases by a value
    determined by multiplying the desired number of bits with the largest value.

    Parameters
    ----------
    network : MultiClassNeuralNetwork
        the neural network to be quantized
    power : int
        bits of the resulting data-types
    array_type : type, optional
        numpy type of the resulting network

    Returns
    -------
    out : MultiClassNeuralNetwork
        the quantized model
    """
    s = 2 ** power - 1
    discretized_network = copy.deepcopy(network)
    for layer in discretized_network.layers:
        max_weight = np.max(np.abs(layer.weights))
        # max_weight = np.max(layer.weights)
        # min_weight = np.min(layer.weights)
        scale_weights = s / max_weight
        # zero_weights = 255 - max_weight * scale_weights
        layer.weights = np.round(layer.weights * scale_weights)
        layer.weights = layer.weights.astype(array_type)
        max_bias = np.max(np.abs(layer.biases))
        # max_bias = np.max(layer.biases)
        min_bias = np.min(layer.biases)
        scale_biases = s / max_bias
        # zero_biases = 255 - max_bias*scale_biases
        layer.biases = np.round(layer.biases * scale_biases)
        layer.biases = layer.biases.astype(array_type)
    return discretized_network


def get_binary_network(network):
    """Binary Quantization.

    Implementation of get_quantized_network with power=1.

    Parameters
    ----------
    network : MultiClassNeuralNetwork
        the neural network to be binarized

    Returns
    -------
    out : MultiClassNeuralNetwork
        the binarized model
    """
    discretized_network = copy.deepcopy(network)
    for layer in discretized_network.layers:
        layer.weights = np.where(layer.weights >= 0, 1, 0)
        layer.biases = np.where(layer.biases >= 0, 1, 0)
    return discretized_network
