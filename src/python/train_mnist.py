import numpy as np

from model.neural_network import MultiClassNeuralNetwork
from utils.dataset_util import load_mnist
from utils.neural_network_util import save_model
from utils.discretization_util import get_quantized_network

"""Train MNIST model"""
x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()

n = MultiClassNeuralNetwork([784, 300, 10], ['tanh', 'softmax'], learning_rate=0.003)
n.train(x_train, y_train, x_val, y_val, 100)

print('Test accuracy:', np.mean(n.predict(x_test) == y_test))

# save_model(n, 'mnist', 'tanh_')
# save_model(n, 'mnist', 'tanh_30_')