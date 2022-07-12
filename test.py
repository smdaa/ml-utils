import numpy as np
import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

model = create_model()
model.load_weights('./model.h5')

model_prun = create_model()
model_prun.load_weights('./model_prun.h5')

model_nonzero = 0
model_prun_nonzero = 0

for w in (model.get_weights()):
    model_nonzero = model_nonzero + np.count_nonzero(w)

for w in (model_prun.get_weights()):
    model_prun_nonzero = model_prun_nonzero + np.count_nonzero(w)

print(model_nonzero)
print(model_prun_nonzero)
