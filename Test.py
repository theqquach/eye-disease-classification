import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib
import keras

img_size = (512, 512)
batch_size = 32

data_dir = "dataset/normal"
filenames = tf.constant([os.path.join(data_dir, fname) for fname in os.listdir(data_dir)])
dataset = tf.data.Dataset.from_tensor_slices((filenames))

def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

print(dataset)





