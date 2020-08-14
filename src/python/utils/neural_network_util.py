import numpy as np

"""Collections of functions to manipulate a neural network"""


def save_model(network, dataset, base_name=''):
    """Save model.

    Parameters
    ----------
    network : MultiClassNeuralNetwork
        the network to be saved
    dataset : str
        the dataset used
    base_name : str, optional
        the name of the file

    Returns
    -------
        None
    """
    base_name = '../../resources/' + dataset + '/weights_and_biases/' + base_name
    weights = []
    biases = []
    for layer in network.layers:
        weights.append(layer.weights)
        biases.append(layer.biases)
    weights = np.asarray(weights)
    biases = np.asarray(biases)
    np.save(base_name + 'weights.npy', weights, allow_pickle=True)
    np.save(base_name + 'biases.npy', biases, allow_pickle=True)


def load_model(network, dataset, base_name=''):
    """Load model.

    Parameters
    ----------
    network : MultiClassNeuralNetwork
        the network to be saved
    dataset : str
        the dataset used
    base_name : str, optional
        the name of the file

    Returns
    -------
    out : MultiClassNeuralNetwork
        the loaded network
    """
    base_name = '../../resources/' + dataset + '/weights_and_biases/' + base_name
    weights = np.load(base_name + 'weights.npy', allow_pickle=True)
    biases = np.load(base_name + 'biases.npy', allow_pickle=True)
    layers = network.layers
    for i in range(len(layers)):
        layers[i].weights = weights[i]
        layers[i].biases = biases[i]
    return network


def save_layers_individually(network, dataset, base_name=''):
    """Save the layers of a model in separate files.

    Save the layers in separate files, numbered based on precedence in the network.
    The layers are saved both as .npy files and .csv files.

    Parameters
    ----------
    network : MultiClassNeuralNetwork
        the network to be saved
    dataset : str
        the dataset used
    base_name : str, optional
        the name of the file

    Returns
    -------
        None
    """
    base_name = '../../resources/' + dataset + '/weights_and_biases/' + base_name
    for i in range(len(network.layers)):
        np.savetxt(base_name + 'weights' + str(i) + '.csv', network.layers[i].weights.T, delimiter=',', fmt='%d')
        np.savetxt(base_name + 'biases' + str(i) + '.csv', network.layers[i].biases.T, delimiter=',', fmt='%d')
        np.save(base_name + 'weights' + str(i) + '.npy', network.layers[i].weights, allow_pickle=False)
        np.save(base_name + 'biases' + str(i) + '.npy', network.layers[i].biases, allow_pickle=False)
