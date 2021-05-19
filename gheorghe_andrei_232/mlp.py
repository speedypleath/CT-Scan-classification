from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt  
from readwrite import read_data, write_data
from sklearn.metrics import plot_confusion_matrix

train_images, train_labels, test_images, validation_images, validation_labels = read_data()
classifier = MLPClassifier(hidden_layer_sizes=(100, ),
                            activation='relu', solver='adam', 
                            alpha=0.0001, batch_size='auto', 
                            learning_rate='constant', 
                            power_t=0.5, max_iter=10, 
                            shuffle=True, random_state=None, 
                            tol=0.0001, momentum=0.9, 
                            early_stopping=False, 
                            validation_fraction=0.1, 
                            n_iter_no_change=10)

classifier.fit(train_images, train_labels)

plot_confusion_matrix(classifier, validation_images, validation_labels)
plt.show()
plt.savefig('mlp.png')

prediction = classifier.predict(test_images)
write_data(prediction)