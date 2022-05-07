import numpy as np
from sklearn.utils import shuffle

train_images = np.load('train_images.npy')
test_images  = np.load('test_images.npy')

train_labels = np.load('train_labels.npy')
test_labels  = np.load('test_labels.npy')

x = np.concatenate((train_images, test_images), axis=0)
y = np.concatenate((train_labels, test_labels), axis=0).reshape(-1)

x = x.reshape((x.shape[0], 28, 28))

x, y = shuffle(x, y)

x = x / 255.0

np.save('x.npy', x)
np.save('y.npy', y)