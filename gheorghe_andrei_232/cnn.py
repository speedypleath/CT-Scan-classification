# import numpy as np
# import matplotlib.pyplot as plt
# from keras.preprocessing import image
import time
import keras
from keras import activations
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from readwrite import unpickle_train_data, unpickle_validation_data, write_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
tensorboard = TensorBoard(log_dir="../logs/final2")
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
X_train, Y_train = unpickle_train_data()
print(X_train.shape, Y_train.shape)
X_validate, Y_validate = unpickle_validation_data()
X_train = (X_train / 255.0) - 0.5
X_validate = (X_validate / 255.0) - 0.5


model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(50, 50, 1), activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation="softmax"))


model.summary()

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_validate)
model.fit(X_train, Y_train, epochs=10, validation_split=0.1,
          validation_data=(X_validate, Y_validate), callbacks=[tensorboard])


model.save("CNN.model")
