from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adam

from keras.datasets import cifar10
from keras.utils import np_utils
import matplotlib.pyplot as plt


# CIFAR 10 is a set of 60000 images (32x32x3)
IMG_ROWS, IMG_COLS, IMG_CHAS = 32, 32, 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_CHAS)

# network parameters
NUM_EPOCH = 20
NUM_CLASSES = 10
BATCH_SIZE = 128
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIMIZER = RMSprop()

# shuffled and split between train and test
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# normalize the pixel data to [0, 1] by /255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class
Y_train = np_utils.to_categorical(Y_train, NUM_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)

# build network topology
model = Sequential()
# Conv -> ReLU -> Pool (Dropout)
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Flattern -> Dense -> ReLu (Dropout)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Softmax
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))
model.summary()

# compile model, train and test
model.compile(loss='categorical_crossentropy', 
    optimizer=OPTIMIZER, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
    epochs=NUM_EPOCH, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
score = model.evaluate(X_test, Y_test, 
    batch_size=BATCH_SIZE, verbose=VERBOSE)
print('test score:', score[0])
print('test accuracy:', score[1])

# save model
model_json = model.to_json()
open('cifar10_archi.json', 'w').write(model_json)
model.save_weights('cifar10_weights.h5', overwrite=True)
