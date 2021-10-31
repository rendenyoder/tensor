from abc import ABC
from enum import Enum
from random import randint
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from cli import ModelClient


class EnglishLabel(Enum):
    """
    Enum for the english names of the labels 0-9.
    """
    TShirt = 0
    Pants = 1
    Pullover = 2
    Dress = 3
    Coat = 4
    Sandal = 5
    Shirt = 6
    Sneaker = 7
    Bag = 8
    AnkleBoot = 9


class FashionMnist(ModelClient, ABC):
    """
    Wrapping class for a model based on the fashion_mnist dataset.
    """
    def __init__(self):
        self.__dataset__ = keras.datasets.fashion_mnist.load_data()
        super().__init__()

    def __train_data__(self):
        """
        Gets the training data from the fashion_mnist dataset.
        :return: tuple
        """
        return self.__dataset__[0]

    def __test_data__(self):
        """
        Gets the testing data from the fashion_mnist dataset.
        :return: tuple
        """
        return self.__dataset__[1]

    def __model_from_path__(self):
        """
        Loads a pre-built model.
        :return: model
        """
        return tf.keras.models.load_model('models/fashion_mnist/')

    def __build_and_save__(self):
        """
        Builds the model using the train data set.
        :return: model
        """
        model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                                input_shape=(28, 28, 1)),
            keras.layers.MaxPool2D(2, 2),
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(1024, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        (train_data, train_labels) = self.__train_data__()
        train_data = train_data.reshape(60000, 28, 28, 1)

        model.fit(train_data, train_labels, epochs=10, verbose=1)
        model.save("models/fashion_mnist")
        return model

    def predict(self, image_index):
        """
        Predicts the label for the test image at the given index.
        :param image_index: Index of test image
        """
        (test_data, _) = self.__test_data__()

        plt.figure()
        plt.imshow(test_data[image_index])
        plt.show()

        test_image = test_data[image_index].reshape(1, 28, 28, 1)
        prediction = self.model.predict(test_image)
        max_index = prediction.argmax(axis=-1)
        print(EnglishLabel(max_index[0]))


if __name__ == "__main__":
    fashion_mnist = FashionMnist()
    fashion_mnist.predict(randint(0, 10000))
