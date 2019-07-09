from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adam


# define the ConvNet
class LeNet:

    @staticmethod
    def build(input_shape, classes):
        model = Sequential()

        # Conv -> ReLU -> Pool
        model.add(Conv2D(filters=20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Conv -> ReLU -> Pool
        model.add(Conv2D(filters=50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        # Flattern -> ReLU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # Softmax
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model


##### use LeNet model to train and test
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils


# network and training
NUM_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NUM_CLASSES = 10
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

# shuffled and split between train and test
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
f = np.load('/Users/tengsun/.keras/datasets/mnist.npz')
X_train, Y_train = f['x_train'], f['y_train']
X_test, Y_test = f['x_test'], f['y_test']
f.close()

# normalize the pixel data to [0, 1] by /255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# reshape to 60000 x [28 x 28 x 1] as input
X_train = X_train[:, :, :, np.newaxis]
X_test = X_test[:, :, :, np.newaxis]

# convert class vectors to binary class
Y_train = np_utils.to_categorical(Y_train, NUM_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)

# compile model and train by fit
model = LeNet.build(input_shape=INPUT_SHAPE, classes=NUM_CLASSES)
model.compile(loss='categorical_crossentropy', 
    optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
    epochs=NUM_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# evaluate the model using test data
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('test score:', score[0])
print('test accuracy:', score[1])
print(history.history.keys())
