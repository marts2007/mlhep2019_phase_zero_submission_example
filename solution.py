#!/usr/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals
import sys


import tensorflow as tf

def main():
    # print command line arguments
    input_dir, output_dir = sys.argv[1:]

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:

    # In[4]:

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train and evaluate the model:

    # In[5]:

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)

    model.save(output_dir+'/my_model.h5')

    return 0

if __name__ == "__main__":
    main()
