
import matplotlib.pyplot as plt  
from readwrite import read_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from readwrite import read_data, write_data

train_images, train_labels, test_images, validation_images, validation_labels = read_data()

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_images, train_labels)

plot_confusion_matrix(classifier, validation_images, validation_labels)
plt.show()
plt.savefig('knn.png')

prediction = classifier.predict(test_images)
write_data(prediction)