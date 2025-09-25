import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def create_cifar10_cnn(input_shape = (32, 32, 3), num_classes = 10):
  """
  Creating a simple CNN model
  """
  model = Sequential()

  #Convolution block 1
  model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', input_shape = input_shape))
  #Adding maxpooling layer
  model.add(MaxPooling2D((2,2)))

  #Convolution block 2
  model.add(Conv2D(64, (3,3), activation = 'relu', padding = "same"))
  model.add(MaxPooling2D((2,2)))

  #Convolution block 3
  model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

  #Flatten the layer
  model.add(Flatten())

  #Adding hidden or dense layer
  model.add(Dense(64, activation = 'relu'))

  #Final output layer
  model.add(Dense(num_classes, activation = 'softmax'))

  return model
