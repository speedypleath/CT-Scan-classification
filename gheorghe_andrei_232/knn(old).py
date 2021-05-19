from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
    
train_images, train_labels, test_images = [], [], []
for x in open("train.txt").read().split():
    train_images.append(np.asarray(Image.open("train/" + x[:-2]), dtype=np.int8).ravel())
    train_labels.append(int(x[-1]))

train_labels = np.array(train_labels)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_images, train_labels)

nr = 0
n = 0
out = open("out.csv", 'w')
out.write("id,label\n")
# for i, x in enumerate(test_images):
#     print(i)
#     out.write(str(classifier.predict(x.reshape(1, -1))[0]))

for x in open("test.txt").read().split():
    test_images.append(np.asarray(Image.open("test/" + x), dtype=np.int8).ravel())

test_images = np.array(test_images)
prediction = classifier.predict(test_images)

for x, y in zip(open("test.txt").read().split(), prediction):
    out.write(str(x) + "," + str(y) + '\n')