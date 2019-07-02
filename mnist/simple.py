from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
np.random.seed(1671)


# network and training
NUM_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NUM_CLASSES = 10
NUM_HIDDEN = 128
OPTIMIZER = Adam() # SGD, RMSprop
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3
RESHAPED = 784

# shuffled and split between train and test
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
f = np.load('/Users/tengsun/.keras/datasets/mnist.npz')
X_train, Y_train = f['x_train'], f['y_train']
X_test, Y_test = f['x_test'], f['y_test']
f.close()

# X_train is 60000 rows of 28x28=784 values
X_train = X_train.reshape(60000, RESHAPED).astype('float32')
X_test = X_test.reshape(10000, RESHAPED).astype('float32')

# normalize the pixel data to [0, 1] by /255
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class
Y_train = np_utils.to_categorical(Y_train, NUM_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NUM_CLASSES)

# final stage is softmax
model = Sequential()
model.add(Dense(NUM_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NUM_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))
model.summary()

# compile model and train by fit
model.compile(loss='categorical_crossentropy', 
    optimizer=OPTIMIZER, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
    epochs=NUM_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# evaluate the model using test data
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print('test score:', score[0])
print('test accuracy:', score[1])
