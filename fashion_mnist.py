import os.path
from enum import Enum
from random import randint
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


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


class FashionMnist:
    """
    Wrapping class for a model based on the fashion_mnist dataset.
    """
    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist
        dataset = fashion_mnist.load_data()
        (self.train_images, self.train_labels) = dataset[0]
        (self.test_images, self.test_labels) = dataset[1]
        self.model = self.load_model()

    def load_model(self):
        path = "models/fashion_mnist"

        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
        else:
            model = keras.Sequential([
                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                                    input_shape=(28, 28, 1)),
                keras.layers.MaxPool2D(2, 2),
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(1024, activation=tf.nn.relu),
                keras.layers.Dense(10, activation=tf.nn.softmax)
            ])

            self.train_images = self.train_images.reshape(60000, 28, 28, 1)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model.fit(self.train_images, self.train_labels, epochs=10)

            model.save("models/fashion_mnist")

        return model

    def predict(self, image_index):
        plt.figure()
        plt.imshow(self.test_images[image_index])
        plt.show()

        test_image = self.test_images[image_index].reshape(1, 28, 28, 1)
        classifications = self.model.predict(test_image)
        classification = classifications[0]
        max_val = np.max(classification)
        most_probable = np.where(classification == max_val)
        return EnglishLabel(most_probable[0][0])


def main():
    fashion_mnist = FashionMnist()
    random = randint(0, 10000)
    label = fashion_mnist.predict(random)
    print(label)


if __name__ == "__main__":
    main()
