from abc import ABC
from cli import ModelClient
import numpy
import tensorflow as tf
from matplotlib import image
from keras_preprocessing.image import ImageDataGenerator
import requests
import zipfile
import io
import os


def __load_data__(url, path):
    """
    Downloads data using the url and creates a data generator from it.
    :return: ImageDataGenerator
    """
    if not os.path.exists(path):
        print('Downloading dataset from {}'.format(url))
        req = requests.get(url)
        content = zipfile.ZipFile(io.BytesIO(req.content))
        content.extractall(path)

    data_gen = ImageDataGenerator(rescale=1.0 / 255)
    generator = data_gen.flow_from_directory(
        directory=path,
        target_size=(300, 300),
        class_mode='categorical'
    )
    return generator


class Roshambo(ModelClient, ABC):
    """
    Builds a model based on the rock_paper_scissors dataset available
    at https://laurencemoroney.com/rock-paper-scissors-dataset.
    """
    def __init__(self):
        super().__init__()

    def __train_data__(self):
        """
        Downloads the training data and creates a generator from it.
        :return: ImageDataGenerator
        """
        url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'
        return __load_data__(url, 'data/train/rps/')

    def __test_data__(self):
        """
        Downloads the testing data and creates a generator from it.
        :return: ImageDataGenerator
        """
        url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip'
        return __load_data__(url, 'data/test/rps-test-set/')

    def __model_from_path__(self):
        """
        Loads a pre-built model.
        :return: model
        """
        return tf.keras.models.load_model('models/roshambo/')

    def __build_and_save__(self):
        """
        Builds the model using the train/test data set.
        :return: model
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                   input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy']
        )

        train = self.__train_data__()
        test = self.__test_data__()

        model.fit(train, epochs=25, validation_data=test, verbose=1)
        model.save('models/roshambo')
        return model

    def predict(self, content):
        """
        Prints out the predicted label for content.
        :param content: A ndarray derived from 300 x 300 image.
        """
        if type(content) is not numpy.ndarray:
            print('Content must be a ndarray.')
            return

        if content.shape != (300, 300, 4) and content.shape != (300, 300, 3):
            print("Invalid input shape {0}".format(content.shape))
            return

        if content.shape == (300, 300, 4):
            no_alpha = content[:, :, :3]
            reshaped = no_alpha.reshape(1, 300, 300, 3)
        else:
            reshaped = content.reshape(1, 300, 300, 3)

        labels = ['Paper 📄', 'Rock 🪨', 'Scissors ✂️']
        prediction = self.model.predict(reshaped)
        max_index = prediction.argmax(axis=-1)
        print(labels[max_index[0]])


if __name__ == '__main__':
    cli = Roshambo()

    sample = image.imread('data/test/rps-test-set/rock/testrock01-00.png')
    cli.predict(sample)
    sample = image.imread('data/test/rps-test-set/scissors/testscissors01-00.png')
    cli.predict(sample)
    sample = image.imread('data/test/rps-test-set/paper/testpaper01-00.png')
    cli.predict(sample)
