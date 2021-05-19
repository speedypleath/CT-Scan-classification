from keras.backend import dropout
from tensorflow.keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from readwrite import unpickle_train_data, unpickle_validation_data
from tensorflow.keras.utils import to_categorical
import time
import keras
import tensorflow as tf

tensorboard = TensorBoard(log_dir="../logs/NoDense" + str(time.time()))
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
X_train, Y_train = unpickle_train_data()
print(X_train.shape, Y_train.shape)
X_validate, Y_validate = unpickle_validation_data()
X_train = (X_train / 255.0) - 0.5
X_validate = (X_validate / 255.0) - 0.5
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_validate)

dense = [0, 1, 2]
sizes = [64, 128]
conv = [1, 2, 3]
learning_rates = [0.001, 0.0001]
dropouts = [0.1, 0.2, 0.3]
for x in dense:
    for y in sizes:
        for z in conv:
            for rate in learning_rates:
                tensorboard = TensorBoard(log_dir="../logs/{}-conv-{}-nodes-{}-dense-rate{}-{}".format(x, y, z, rate, time.time()))
                model = Sequential()

                model.add(Conv2D(64, (3, 3), input_shape=(50, 50, 1), activation="relu"))
                model.add(MaxPooling2D((2, 2)))

                for i in range(z - 1):
                    model.add(Conv2D(y, (3, 3), activation='relu'))
                    model.add(MaxPooling2D((2, 2)))

                model.add(Flatten())
                
                for i in range(x):
                    model.add(Dense(512, activation='relu'))
                    
                model.add(Dense(3, activation="softmax"))


                model.summary()

                opt = keras.optimizers.Adam(learning_rate=rate)
                model.compile(optimizer=opt,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
                
                model.fit(X_train,
                          Y_train,
                          epochs=10,
                          validation_split=0.1,
                          validation_data=(X_validate, Y_validate), 
                          callbacks=[tensorboard])
