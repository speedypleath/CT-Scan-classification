import keras
from readwrite import unpickle_test_data, unpickle_validation_data, write_data
from PIL import Image
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
X_validation, Y_validation = unpickle_validation_data()
model = keras.models.load_model("CNN.model")

X_validate, Y_validate = unpickle_validation_data()

X_test = unpickle_test_data()
X_test = (X_test / 255.0) - 0.5
X_validate = (X_validate / 255.0) - 0.5
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

write_data(predictions)

# nr = 0
# for i, x in enumerate(predictions):
#     print(x, Y_validation[i])
#     print(nr / i if i>0 else 1)
#     if x == Y_validation[i]:
#         nr += 1
# prediction = model.predict(X_validation.reshape(len(X_validation), 50, 50 , 1))

# nr = 0
# for i, x in enumerate(prediction):
#     print(x, Y_validation[i])
#     print(nr / i if i>0 else 1)
#     if np.where(x == 1) == Y_validation[i]:
#         nr += 1
