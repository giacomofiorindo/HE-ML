import numpy as np

from model.activation import get_activation_functions


class Layer(object):
    """General class for a feed-forward neural network layer.

    The class implements the base functionalities of a fully connected layer.
    Namely forward and back-propagation methods with fixed learning rate.
    """

    def __init__(self, input_units, output_units, activation_function='relu', learning_rate=0.0003):
        """Layer constructor.

        Constructor of a feed-forward layer. Weights and biases are initialised randomly
        and normalized using a small number. The learning rate is fixed and does not change
        during the learning procedure.

        Parameters
        ----------
        input_units : int
            number of inputs.
        output_units : int
            number of neurons in the layer.
        activation_function: str, optional
            the desired activation function
        learning_rate: float, optional
            the learning rate
        """
        self._learning_rate = learning_rate

        # initialize weights and biases with small random numbers. We use normal initialization
        self._weights = np.random.randn(input_units, output_units) * 0.01
        self._biases = np.random.randn(output_units) * 0.01

        self._output = np.zeros(output_units)
        self._test = np.zeros(output_units)
        self._activation_function, self._derivative_activation_function = get_activation_functions(activation_function)

    def forward(self, inputs):
        """ Forward method.

        Parameters
        ----------
        inputs : ndarray
            2D array containing the input values to the layer.

        Returns
        -------
        out : ndarray
            2D array containing the output values of the layer.

        """
        self._test = np.dot(inputs, self._weights) + self._biases
        # Separate line for testing purposes
        self._output = self._activation_function(self._test)
        return self._output

    # delta_error -> error of next layer
    def backward(self, inputs, delta_error, last=False):
        """Back-propagation method.

        Parameters
        ----------
        inputs : ndarray
            2D array containing the input values to the layer.
        delta_error : ndarray
            2D array containing the back-propagate error of the following layer.
        last : bool, optional
            flag that indicates if the layer is the last one in the network.

        Returns
        -------
        out : ndarray
            2D array containing the error of the current layer.
        """
        if not last:
            delta_error = delta_error * self._derivative_activation_function(self._output)

        # compute gradient w.r.t. weights and biases
        grad_weights = inputs.T.dot(delta_error)
        grad_biases = np.sum(delta_error, axis=0)

        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self._weights = self._weights - self._learning_rate * grad_weights
        self._biases = self._biases - self._learning_rate * grad_biases

        # error on the input
        delta_error = delta_error.dot(self._weights.T)
        return delta_error

    @property
    def weights(self):
        """ndarray: the weights of the layer"""
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights

    @property
    def biases(self):
        """ndarray: the biases of the layer"""
        return self._biases

    @biases.setter
    def biases(self, new_biases):
        self._biases = new_biases