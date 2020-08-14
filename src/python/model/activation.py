import numpy as np

"""Collection of activation functions and their derivatives"""


def get_activation_functions(activation_function):
    """
    Return the activation and derivative functions corresponding to the inputted string. If the activation function is
    not supported default functions are returned instead.

    Parameters
    ----------
    activation_function : str
        the desired activation function

    Returns
    -------
    (function, function)
        the activation function, the derivative function

    """
    return _get_activation(activation_function), _get_derivative_activation(activation_function)


def _get_activation(activation_function):
    """return the desired activation function"""
    if activation_function == 'relu':
        return _relu
    if activation_function == 'relu6':
        return _relu6
    if activation_function == 'leaky-relu':
        return _leaky_relu
    if activation_function == 'softmax':
        return _softmax
    if activation_function == 'sign':
        return _sign
    if activation_function == 'hard-sigmoid':
        return _hard_sigmoid
    if activation_function == 'hard-clip':
        return _hard_clip
    if activation_function == 'tanh':
        return _tanh
    # default case
    return _neutral_activation


def _get_derivative_activation(activation_function):
    """return the desired derivative function"""
    if activation_function == 'relu':
        return _grad_relu
    if activation_function == 'leaky-relu':
        return _grad_leaky_relu
    if activation_function == 'relu6':
        return _grad_relu6
    if activation_function == 'hard-sigmoid':
        return _grad_hard_sigmoid
    if activation_function == 'hard-clip':
        return _grad_hard_clip
    if activation_function == 'tanh':
        return _grad_tanh
    # default case
    return _neutral_activation


def _relu(x):
    """Implementation of ReLU"""
    return np.maximum(0, x)


def _relu6(x):
    """Implementation of ReLU6"""
    return np.minimum(np.maximum(0, x), 6)


def _leaky_relu(x):
    """Implementation of Leaky ReLU"""
    return np.where(x > 0, x, x * 0.01)


def _softmax(x):
    """Safe implementation of Softmax"""
    # exps = np.exp(x)
    # return exps / np.sum(exps, axis=1, keepdims=True)
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def _sign(x):
    """Implementation of Sign"""
    # return np.sign(x)
    return np.where(x >= 0, 1, -1)
    # return np.where(x >= 0, 1, 0)


def _tanh(x):
    """Implementation of tanh"""
    return np.tanh(x)


def _hard_sigmoid(x):
    """Implementation of Hard Sigmoid"""
    return np.clip(0.2 * x + 0.5, 0, 1)


def _hard_clip(x):
    """Implementation of Hard Clip"""
    return np.clip(x, -1, 1)


def _neutral_activation(x):
    """Neutral activation - input is not changed"""
    return x


def _grad_relu(x):
    """Derivative of ReLU"""
    relu_grad = x > 0
    return relu_grad


def _grad_relu6(x):
    """Derivative of ReLU6"""
    relu_grad = np.ones_like(x)
    relu_grad[x < 0] = 0
    relu_grad[x > 6] = 0
    return relu_grad


def _grad_leaky_relu(x):
    """Derivative of Leaky ReLU"""
    relu_grad = np.ones_like(x)
    relu_grad[x < 0] = 0.01
    return relu_grad


def _grad_softmax(delta):
    """Derivative of Softamax"""
    return delta


def _grad_hard_sigmoid(x):
    """Derivative of Hard Sigmoid"""
    np.where((-2.5 < x) & (x < 2.5), 0.2, 0)
    return x


def _grad_hard_clip(x):
    """Derivative of Hard Clip"""
    np.where((-1 <= x) & (x <= 1), 1, 0)
    return x


def _grad_tanh(x):
    """Derivative of tanh"""
    return 1.0 - np.tanh(x) ** 2
