import numpy as np

from model.neural_network import MultiClassNeuralNetwork
from utils.dataset_util import load_mnist
from utils.discretization_util import get_naive_discretized_network, get_quantized_network
from utils.neural_network_util import load_model, save_layers_individually

"""Compare original model with discretized versions for MNIST dataset"""

original_tanh = MultiClassNeuralNetwork([784, 300, 10], ['tanh', 'softmax'])
original_tanh = load_model(original_tanh, 'mnist', 'tanh_')

original_tanh_30 = MultiClassNeuralNetwork([784, 30, 10], ['tanh', 'softmax'])
original_tanh_30 = load_model(original_tanh_30, 'mnist', 'tanh_30_')

naive_sign_tanh = MultiClassNeuralNetwork([784, 300, 10], ['sign', ''])
naive_sign_tanh = load_model(naive_sign_tanh, 'mnist', 'tanh_')
naive_sign_tanh = get_naive_discretized_network(naive_sign_tanh)

quantized_sign = MultiClassNeuralNetwork([784, 300, 10], ['sign', ''])
quantized_sign = load_model(quantized_sign, 'mnist', 'tanh_')
quantized_sign = get_quantized_network(quantized_sign, 4)

quantized_sign_30 = MultiClassNeuralNetwork([784, 30, 10], ['sign', ''])
quantized_sign_30 = load_model(quantized_sign_30, 'mnist', 'tanh_30_')
quantized_sign_30 = get_quantized_network(quantized_sign_30, 4)

naive_sign_tanh_30 = MultiClassNeuralNetwork([784, 30, 10], ['sign', ''])
naive_sign_tanh_30 = load_model(naive_sign_tanh_30, 'mnist', 'tanh_30_')
naive_sign_tanh_30 = get_naive_discretized_network(naive_sign_tanh_30)

# save_layers_individually(quantized_sign, 'mnist', 'q_')

_, _, _, _, x_test, y_test = load_mnist(round_dataset=True)

print('Test accuracy original (tanH):', np.mean(original_tanh.predict(x_test) == y_test))
print('Test accuracy quantized (sign):', np.mean(quantized_sign.predict(x_test) == y_test))
print('Test accuracy naive (sign) from tanH:', np.mean(naive_sign_tanh.predict(x_test) == y_test))
print('Test accuracy original (TanH 30):', np.mean(original_tanh_30.predict(x_test) == y_test))
print('Test accuracy quantized (sign 30):', np.mean(quantized_sign_30.predict(x_test) == y_test))
print('Test accuracy naive (sign 30):', np.mean(naive_sign_tanh_30.predict(x_test) == y_test))

'''
Test accuracy original (ReLU): 0.9415
Test accuracy original (Hard Sigmoid): 0.8545
Test accuracy naive (ReLU) from ReLU: 0.938
Test accuracy naive (ReLU) from Hard Sigmoid: 0.848
Test accuracy naive (sign) from ReLU: 0.6794
Test accuracy naive (sign) from Hard Sigmoid: 0.8363
Test accuracy quantized (ReLU): 0.9417
Test accuracy quantized (sign): 0.8493
'''
