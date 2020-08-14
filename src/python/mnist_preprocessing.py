import keras
import numpy as np

"""Load and save the MNIST dataset in .npy and .csv format. The dataset is also normalized"""
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
x_train = x_train.astype(float) / 255.
x_test = x_test.astype(float) / 255.

np.save('../../resources/mnist/input_mnist_train.npy', x_train)
np.save('../../resources/mnist/labels_mnist_train.npy', y_train)
np.save('../../resources/mnist/input_mnist_test.npy', x_test)
np.save('../../resources/mnist/labels_mnist_test.npy', y_test)
np.savetxt('../../resources/mnist/input_mnist_train.csv', np.round(x_train), delimiter=',', fmt='%d')
np.savetxt('../../resources/mnist/labels_mnist_train.csv', y_train, delimiter=',', fmt='%d')
np.savetxt('../../resources/mnist/input_mnist_test.csv', np.round(x_test), delimiter=',', fmt='%d')
np.savetxt('../../resources/mnist/labels_mnist_test.csv', y_test, delimiter=',', fmt='%d')
