import numpy as np

"""Collection of functions to load and manipulate the MNIST and Breast Cancer Wisconsin datasets"""


def load_mnist(round_dataset=False):
    """Load the MNIST dataset.

    Load the MNIST dataset from the 'resources' folder of the project. The dataset is assumed to
    be there. The dataset is normalised and can be rounded.

    Parameters
    ----------
    round_dataset : bool
        True if the database needs to be rounded, false otherwise.

    Returns
    -------
    (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray)
        The dataset split in training input, training target,
        validation input, validation target, test input, test target.
    """
    x_train = np.load('../../resources/mnist/input_mnist_train.npy').astype(np.float32)
    y_train = np.load('../../resources/mnist/labels_mnist_train.npy').astype(np.int32)
    x_test = np.load('../../resources/mnist/input_mnist_test.npy').astype(np.float32)
    y_test = np.load('../../resources/mnist/labels_mnist_test.npy').astype(np.int32)

    if round_dataset:
        x_train = np.round(x_train)
        x_test = np.round(x_test)
    # we reserve the last 10000 training examples for validation
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_tumor_stats():
    """Load the Breast Cancer Wisconsin dataset.

    Load the Breast Cancer Wisconsin dataset from the 'resources' folder of the project. The dataset is assumed to
    be there.

    Returns
    -------
    (ndarray, ndarray, ndarray, ndarray, ndarray, ndarray)
        The dataset split in training input, training target,
        validation input, validation target, test input, test target.
    """
    dataset = np.load('../../resources/breast/input_breast.npy').astype(np.float32)
    labels = np.load('../../resources/breast/labels_breast.npy').astype(np.int32)
    train_split = round(dataset.shape[0] * 0.7)
    val_split = round(dataset.shape[0] * 0.8)
    x_train, x_val, x_test = dataset[:train_split], dataset[train_split:val_split], dataset[val_split:]
    y_train, y_val, y_test = labels[:train_split], labels[train_split:val_split], labels[val_split:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    """Batches generator.

    Generator of fixed size batches from two input arrays. The two arrays are batched
    in the same way to guarantee correspondence. The two arrays must be of the same size.

    Parameters
    ----------
    inputs : ndarray
        first array to iterate
    targets : ndarray
        second array to iterate
    batch_size : int
        size of a batch
    shuffle : bool
        True if the batches should follow the same order of the original arrays, False otherwise.

    Yields
    ------
    (ndarray, ndarray)
        The next batches for the arrays.

    Examples
    --------
    >>> a = np.random.randint(5, size=5)
    array([1, 0, 4, 2, 4])
    >>> b = np.random.randint(5, size=5)
    array([3, 2, 2, 1, 1])
    >>> print([i for i in iterate_minibatches(a, b, 2)])
    [(array([1, 0]), array([3, 2])), (array([4, 2]), array([2, 1]))]

    """
    assert len(inputs) == len(targets)
    if shuffle:
        random_indices = np.random.permutation(len(inputs))
    for start_index in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            selected_indices = random_indices[start_index:start_index + batch_size]
        else:
            selected_indices = slice(start_index, start_index + batch_size)
        yield inputs[selected_indices], targets[selected_indices]
