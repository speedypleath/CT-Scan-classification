import numpy as np
from PIL import Image
from numpy.core.fromnumeric import ravel
import pickle

def read_training_data():
    training_data = []
    for x in open("../train.txt").read().split():
        img = np.asarray(Image.open("../train/" + x[:-2]), dtype=np.int8)
        training_data.append((img, int(x[-1])))
    return training_data

def read_validation_data():
    validation_data = []
    for x in open("../train.txt").read().split():
        img = np.asarray(Image.open("../train/" + x[:-2]), dtype=np.int8)
        validation_data.append((img, int(x[-1])))
    return validation_data

def read_test_data():
    test_data = []
    for x in open("../test.txt").read().split():
        img = np.asarray(Image.open("../test/" + x), dtype=np.int8)
        test_data.append(img)
    return test_data

def read_data(flat=True):
    train_images = []
    train_labels = []
    test_images = []
    validation_images = []
    validation_labels = []
    
    for x in open("../train.txt").read().split():
        img = np.asarray(Image.open("../train/" + x[:-2]), dtype=np.int8)
        img = img.ravel() if flat else img
        train_images.append(img)
        train_labels.append(int(x[-1]))
        
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    validation_images = []
    validation_labels = []
    for x in open("../validation.txt").read().split():
        img = np.asarray(Image.open("../validation/" + x[:-2]), dtype=np.int8)
        img = img.ravel() if flat else img #
        validation_images.append(img)
        validation_labels.append(int(x[-1]))
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)

    for x in open("../test.txt").read().split():
        img = np.asarray(Image.open("../test/" + x), dtype=np.int8)
        img = img.ravel() if flat else img
        test_images.append(img)
    test_images = np.array(test_images)
    
    return train_images, train_labels, test_images, validation_images, validation_labels

def write_data(prediction):
    out = open("../out.csv", 'w')
    out.write("id,label\n")
    for x, y in zip(open("../test.txt").read().split(), prediction):
        out.write(str(x) + "," + str(y) + '\n')
        
def pickle_data(training_data, validation_data, test_data):
    X = []
    Y = []
    
    for x, y in training_data:
        X.append(x)
        Y.append(y)
        
    X = np.array(X).reshape(-1, 50, 50, 1)
    # 50x50 grayscale 
    out = open("../pickles/X_train.pickle", 'wb')
    pickle.dump(X, out)
    out.close()
    
    out = open("../pickles/Y_train.pickle", 'wb')
    pickle.dump(Y, out)
    out.close()
    
    
    X = []
    Y = []
    
    for x, y in validation_data:
        X.append(x)
        Y.append(y)
        
    X = np.array(X).reshape(-1, 50, 50, 1)
    
    out = open("../pickles/X_validation.pickle", 'wb')
    pickle.dump(X, out)
    out.close()
    
    out = open("../pickles/Y_validation.pickle", 'wb')
    pickle.dump(Y, out)
    out.close()
    
    X = []
    
    for x in test_data:
        X.append(x)
        
    X = np.array(X).reshape(-1, 50, 50, 1)
    
    out = open("../pickles/X_test.pickle", 'wb')
    pickle.dump(X, out)
    out.close()
    
def unpickle_test_data():
    f = open("../pickles/X_test.pickle", 'rb')
    X_test = pickle.load(f)
    X_test = np.array(X_test)
    return(X_test)
    
def unpickle_validation_data():
    f = open("../pickles/X_validation.pickle", 'rb')
    X_validation = pickle.load(f)
    X_validation = np.array(X_validation)
    
    f = open("../pickles/Y_validation.pickle", 'rb')
    Y_validation = pickle.load(f)
    Y_validation = np.array(Y_validation)
    
    return X_validation, Y_validation
    
def unpickle_train_data():
    f = open("../pickles/X_train.pickle", 'rb')
    X_train = pickle.load(f)
    X_train = np.array(X_train)
    
    f = open("../pickles/Y_train.pickle", 'rb')
    Y_train = pickle.load(f)
    Y_train = np.array(Y_train)
    
    return X_train, Y_train

def main():
    training_data = read_training_data()
    validation_data = read_validation_data()
    test_data = read_test_data()
    pickle_data(training_data, validation_data, test_data)

if __name__ == '__main__':
    main()
    