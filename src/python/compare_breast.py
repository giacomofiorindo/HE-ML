import numpy as np

from model.neural_network import MultiClassNeuralNetwork
from utils.dataset_util import load_tumor_stats
from utils.discretization_util import get_naive_discretized_network, get_quantized_network
from utils.neural_network_util import load_model, save_layers_individually

"""Compare original model with discretized versions for Breast Cancer Wisconsin dataset"""

original_tanh = MultiClassNeuralNetwork([30, 40, 20, 2], ['tanh', 'tanh', 'softmax'])
original_tanh = load_model(original_tanh, 'breast', 'breast_tanh_')

naive_sign_tanh = MultiClassNeuralNetwork([30, 40, 20, 2], ['sign', 'sign', ''])
naive_sign_tanh = load_model(naive_sign_tanh, 'breast', 'breast_tanh_')
naive_sign_tanh = get_naive_discretized_network(naive_sign_tanh)

quantized_sign = MultiClassNeuralNetwork([30, 40, 20, 2], ['sign', 'sign', ''])
quantized_sign = load_model(quantized_sign, 'breast', 'breast_tanh_')
quantized_sign = get_quantized_network(quantized_sign, 2)

# save_layers_individually(quantized_sign, 'breast', 'q_breast_')

_, _, _, _, x_test, y_test = load_tumor_stats()
# x_test = np.round(x_test/20)

print('Test accuracy original (tanH):', np.mean(original_tanh.predict(x_test) == y_test))
print('Test accuracy naive (sign) from tanH:', np.mean(naive_sign_tanh.predict(x_test) == y_test))
print('Test accuracy quantized (sign):', np.mean(quantized_sign.predict(x_test) == y_test))
