import numpy as np

from model.neural_network import MultiClassNeuralNetwork
from utils.dataset_util import load_tumor_stats
from utils.neural_network_util import save_model

"""Train Breast Cancer Wisconsin model"""
x_train, y_train, x_val, y_val, x_test, y_test = load_tumor_stats()

n = MultiClassNeuralNetwork([30, 40, 20, 2], ['tanh', 'tanh', 'softmax'], learning_rate=0.03)
n.train(x_train, y_train, x_val, y_val, 100)

print('Test accuracy:', np.mean(n.predict(x_test) == y_test))

# save_model(n, 'breast', 'breast_tanh_')