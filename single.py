import tensorflow as tf
import numpy as np
from tensorflow import keras


def main():
    """
    Creating and training a simple model containing a single layer
    and a single neuron for the function f(x) = 3x + 1
    :return: None
    """
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

    model.fit(xs, ys, epochs=1000)
    prediction = model.predict([2813.18])

    print(prediction)


if __name__ == '__main__':
    main()
