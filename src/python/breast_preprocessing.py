import numpy as np
import pandas as pd

"""Load and save the Breast Cancer Wisconsin dataset in .npy and .csv format. The dataset is also normalized"""

dataset = pd.read_csv('../../resources/breast/breast_cancer.csv', header=0)

# delete useless columns
dataset.drop('Unnamed: 32', axis=1, inplace=True)
dataset.drop('id', axis=1, inplace=True)
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})

# get the max value for each column
desc = dataset.describe(include='all')
maxs = desc.loc[['max']].values
maxs = np.array(maxs).reshape((31,))
dataset = np.array(dataset)

# normalize each column using the max value and round values
for i in range(len(maxs)):
    '''
    scale_factor = np.round(np.mean(np.log10(np.abs(means[i]) + 1e-10)))
    if scale_factor < 0:
        dataset[:, i] = np.round(dataset[:, i] * 10 ** np.abs(scale_factor))
    if scale_factor > 1:
        dataset[:, i] = np.round(dataset[:, i] * 10 ** (1 - scale_factor))
    else:
        dataset[:, i] = np.round(dataset[:, i])
    '''
    dataset[:, i] = np.round(dataset[:, i] / maxs[i])

np.save('../../resources/breast/input_breast.npy', dataset[:, 1:])
np.save('../../resources/breast/labels_breast.npy', dataset[:, 0])
np.savetxt('../../resources/breast/input_breast.csv', dataset[:, 1:].astype(np.uint8), delimiter=',', fmt='%d')
np.savetxt('../../resources/breast/labels_breast.csv', dataset[:, 0].astype(np.uint8), delimiter=',', fmt='%d')
