from sklearn import preprocessing
import numpy as np
from PIL import Image
train_images, train_labels, test_images = [], [], []
for x in open("train.txt").read().split():
    train_images.append(np.asarray(Image.open("train/" + x[:-2]), dtype=np.int8).ravel())
    train_labels.append(int(x[-1]))
    
train_images = np.array(train_images)
train_labels = np.array(train_labels)

validation_images = []
validation_labels = []
for x in open("validation.txt").read().split():
    validation_images.append(np.asarray(Image.open("validation/" + x[:-2]), dtype=np.int8).ravel())
    validation_labels.append(int(x[-1]))
validation_images = np.array(validation_images)

for x in open("test.txt").read().split():
    test_images.append(np.asarray(Image.open("test/" + x), dtype=np.int8).ravel())

test_images = np.array(test_images)

scaler = preprocessing.StandardScaler()