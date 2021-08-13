from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential



# model taken from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# Ge register the model under the name "cnn_example". Simply use that in the "model" section of the config file.
def cnn_example(nb_channel=32, nb_class=10, dropout=0.5):

    model = Sequential()
    model.add(Conv2D(nb_channel, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    
    model.add(Conv2D(2*nb_channel, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(4*nb_channel, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_class, activation='softmax'))
    
    return model


