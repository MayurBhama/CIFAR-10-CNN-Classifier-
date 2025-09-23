import tensorflow as tf 
from tensorflow import keras 

def get_cifar10_dataset(num_classes = 10):
  """
  Loads and preprocesses the CIFAR-10 dataset.

  Returns: 
  A Tuple of (x_train, y_train), (x_test, y_test).
  Images are normalized to [0,1] and labels are one-hot encoded.
  """

  #Load dataset using keras
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

  #Normalize pixal values from integer [0, 255] to float [0.0, 1.0]
  #Neural network generally perform better with small input values
  x_train = x_train.astype("float32") / 255.0 #original pixel values are stored as uint8 which cannot be stored as decimal values so we convert them to float 32, which can store decimal values like 0.5, 0.235 etc.
  x_test = x_test.astype("float32") / 255.0

  #covert labels to one-hot encoding
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  return (x_train, y_train), (x_test, y_test)
