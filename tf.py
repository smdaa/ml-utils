import time
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def summarize_diagnostics(histories):
	for i in range(len(histories)):
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
	plt.show()

def summarize_performance(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	plt.boxplot(scores)
	plt.show()

# load data
x = np.load('x.npy')
y = np.load('y.npy')

print('working with', y.size, 'samples')

# shuffle data
x, y = shuffle(x, y)

# 5 fold cross validation
n_splits  = 5
n_repeats = 10
scores    = []
histories = []

kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

start = time.time()
for train_index, test_index in kf.split(x):
    # train test split
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # load model 
    model = create_model()
    learning_rate = 0.01
    momentum      = 0.9
    optimizer     = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    loss_fn       = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # train model
    epochs     = 10
    batch_size = 32
    history    = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    histories.append(history)

    # test model
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy : %.3f' % (accuracy * 100.0))
    scores.append(accuracy)

end = time.time()
summarize_diagnostics(histories)
summarize_performance(scores)
print('elapsed time', end - start)
