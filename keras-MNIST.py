import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import argparse


#This is a parser in order to run the program. To run the small network, type "python keras-MNIST.py small". 
#To run the large network, type "python keras-MNIST.py large"
def make_parser():
    parser = argparse.ArgumentParser(description = 'Neural Net Size')
    parser.add_argument('size', help = 'choose the size of the neural network. Enter "small" or "large"')
    return parser

(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

def create_small_model():
    model = Sequential()
    model.add(Dense(32, activation = 'sigmoid', input_shape = (1, 28, 28)))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))

    model.compile(optimizer = "sgd", loss = 'categorical_crossentropy', metrics = ['accuracy'])

    batch_size = 128
    epochs_for_training = 10
    verb = False
    return model, batch_size, epochs_for_training, verb

def create_large_model():
    model = Sequential()
 
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2))) 
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
 
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    batch_size = 32
    epochs_for_training = 10
    verb = 1
    return model, batch_size, epochs_for_training, verb

def evaluate(model, batch, num_epochs, verb):
    results = model.fit(X_train, Y_train, batch_size=batch, epochs = num_epochs, verbose = verb, validation_data = (X_test, Y_test)) 

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=False)
    
    N = np.arange(0, num_epochs)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, results.history["loss"], label="train_loss")
    plt.plot(N, results.history["val_loss"], label="val_loss")
    plt.plot(N, results.history["acc"], label="train_acc")
    plt.plot(N, results.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

    print("Test accuracy: ", test_acc)

def main():
    parser = make_parser()
    options = parser.parse_args()
    if options.size == 'small':
        model, batch, epochs, verb = create_small_model()
        evaluate(model, batch, epochs, verb)
    if options.size == 'large':
        model, batch, epochs, verb = create_large_model()
        evaluate(model, batch, epochs, verb)


if __name__ == '__main__':
    main()
