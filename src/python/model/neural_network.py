import matplotlib.pyplot as plt
import numpy as np

from model.base_layer import Layer
from utils.dataset_util import iterate_minibatches


# TODO: make the error function general
def error_function(output, desired_output):
    """Cross-entropy + softmax derivative function

    Implementation of the derivative of the log cross-entropy loss
    together with softmax activation function.

    Parameters
    ----------
    output : ndarray
        2D array containing the output of the neural network
    desired_output : ndarray
        2D array containing the desired output. i.e. the target output.
    Returns
    -------
    out : ndarray
        the initial error of the network
    """
    num_examples = desired_output.shape[0]
    delta = output
    delta[range(num_examples), desired_output] -= 1
    return delta / num_examples


class MultiClassNeuralNetwork(object):
    """General class for a feed-forward neural network

    This class is responsible for the creation of a multi-class neural network and contains
    all the functionalities to train the model.
    """

    def __init__(self, size_layers, activations, learning_rate=0.001):
        """

        Parameters
        ----------
        size_layers : list[int]
            list containing the number of neurons in each layer
        activations : list[str]
            list containing the names of the activation functions to use in each layer
        learning_rate : float, optional
            the learning rate for all the layers ub the network
        """
        self._size_layers = size_layers
        self._activations = activations
        self._layers = []
        self._learning_rate = learning_rate

        # number of layers should be just 1 more than number of functions
        if len(self._size_layers) != (len(self._activations) + 1):
            return

        # create layers
        for i in range(len(self._activations)):
            self.layers.append(Layer(
                input_units=self._size_layers[i],
                output_units=self._size_layers[i + 1],
                activation_function=self._activations[i],
                learning_rate=self._learning_rate))

    @property
    def layers(self):
        """list[Layer]: list of layers in the network."""
        return self._layers

    def _feedforward(self, data_input):
        """Feedforward procedure"""
        activations = [data_input]
        for i in range(len(self.layers)):
            data_input = self.layers[i].forward(data_input)
            activations.append(data_input)

        # assert len(activations) == len(self.layers)
        return activations

    def _backtrack(self, initial_error, activations):
        """Back-propagation procedure"""
        delta_error = self.layers[len(self.layers) - 1].backward(activations[len(activations) - 2], initial_error,
                                                                 True)
        for i in range(2, len(self._size_layers)):
            delta_error = self.layers[len(self.layers) - i].backward(activations[len(activations) - i - 1], delta_error)

    def _train_batch(self, data_input, data_output):
        activations = self._feedforward(data_input)
        # activations[-1] is final output
        initial_error = error_function(activations[-1], data_output)
        self._backtrack(initial_error, activations)

    def predict(self, x):
        """ Predict the classes of the inputs.

        Parameters
        ----------
        x : ndarray
            input or inputs to classify

        Returns
        -------
        out: int, ndarray
            the class/classes of the input(s)
        """
        prob = self._feedforward(x)[-1]
        return np.argmax(prob, axis=-1)

    def train(self, input_data, target_data, val_data=None, val_target=None, epochs=50, batch_size=10, plot=False):
        """Train procedure

        Train the network using the provided training data. The accuracy of the model is
        printed every iteration. In addition, it is possible to provide validation data and/or
        plot the accuracy.

        Parameters
        ----------
        input_data : ndarray
            2D array containing the training input
        target_data : ndarray
            2D array containing the training target output
        val_data : ndarray, optional
            2D array containing the validation input
        val_target : ndarray, optional
            2D array containing the validation target output
        epochs : int, optional
            number of epochs
        batch_size : int, optional
            size of batch
        plot : bool, optional
            True if the accuracy has to be plotted, False otherwise

        Returns
        -------
            None
        """
        train_log = []
        val_log = []
        for epoch in range(epochs):

            for x_batch, y_batch in iterate_minibatches(input_data, target_data, batch_size=batch_size, shuffle=True):
                self._train_batch(x_batch, y_batch)

            train_log.append(np.mean(self.predict(input_data) == target_data))

            print("Epoch", epoch)
            print("Train accuracy:", train_log[-1])

            if val_data is not None and val_target is not None:
                val_log.append(np.mean(self.predict(val_data) == val_target))
                print("Val accuracy:", val_log[-1])

            if plot:
                plt.plot(train_log, label='train accuracy')
                if val_log:
                    plt.plot(val_log, label='val accuracy')
                plt.legend(loc='best')
                plt.grid()
                plt.show()
