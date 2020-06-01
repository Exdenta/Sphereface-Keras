#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_data, load_train_data, load_short_train_data, load_test_data
from models import get_train_model_cosine, save_model_config
from tensorflow.keras.losses import CosineSimilarity
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy import spatial
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import random
import math
import cv2
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress Tensorflow verbose prints
warnings.simplefilter(action='ignore', category=FutureWarning)


tf.test.is_gpu_available()

# https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
# https://stackoverflow.com/questions/49404993/keras-how-to-use-fit-generator-with-multiple-inputs

models_path = Path.cwd() / 'models' / 'sphereface_20_keras'
data_path = Path.cwd() / 'datasets'
train_dataset_path = data_path / 'CASIA-WebFace-112x96'
test_dataset_path = data_path / 'lfw_112x96'

print("Loading data")
train_dataframe = load_short_train_data(data_path, 4000)
# train_dataframe = load_train_data(data_path)
test_dataframe = load_test_data(data_path)


def preprocess_image(image):
    image = (image - 127.5) / 128
    return image


def train_generator(train_dataframe, batch_size_):
    class_mode_ = "binary"
    generator = ImageDataGenerator(
        preprocessing_function=preprocess_image, validation_split=0.0)

    train_generator_X1 = generator.flow_from_dataframe(
        dataframe=train_dataframe,
        directory=str(train_dataset_path) + "/",
        x_col="fileL",
        y_col="flag",
        subset="training",
        batch_size=batch_size_,
        seed=42,
        shuffle=True,
        class_mode=class_mode_,
        color_mode='rgb',
        target_size=(112, 96))

    train_generator_X2 = generator.flow_from_dataframe(
        dataframe=train_dataframe,
        directory=str(train_dataset_path) + "/",
        x_col="fileR",
        y_col="flag",
        subset="training",
        batch_size=batch_size_,
        seed=42,
        shuffle=True,
        class_mode=class_mode_,
        color_mode='rgb',
        target_size=(112, 96))
    while True:
        X1i = train_generator_X1.next()
        X2i = train_generator_X2.next()
        yield [X1i[0], X2i[0]], X1i[1]


def test_generator(test_dataframe, batch_size):
    class_mode_ = "binary"
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image)
    test_generator_X1 = test_datagen.flow_from_dataframe(
        dataframe=test_dataframe,
        directory=str(test_dataset_path) + "/",
        x_col="fileL",
        y_col="flag",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode=class_mode_,
        color_mode='rgb',
        target_size=(112, 96))

    test_generator_X2 = test_datagen.flow_from_dataframe(
        dataframe=test_dataframe,
        directory=str(test_dataset_path) + "/",
        x_col="fileR",
        y_col="flag",
        batch_size=batch_size,
        seed=42,
        shuffle=False,
        class_mode=class_mode_,
        color_mode='rgb',
        target_size=(112, 96))
    while True:
        X1i = test_generator_X1.next()
        X2i = test_generator_X2.next()
        yield [X1i[0], X2i[0]], X1i[1]


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def custom_loss(yTrue, yPred):
    return K.sum(K.log(yTrue) - K.log(yPred))


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1, 1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print(', lr:', step_decay(len(self.losses)))


sd = []

learning_rate = 0.001
decay_rate = 5e-6
momentum = 0.9
batch_size_ = 128
images_per_epoch_ = 100000

sgd = keras.optimizers.SGD(lr=learning_rate,
                           momentum=momentum,
                           decay=decay_rate,
                           nesterov=False)


def scheduler(epoch, lr):
    momentum = 0.8
    decay_rate = 2e-6
    lr = learning_rate
    return lr


def step_decay(losses):
    lrate = learning_rate
    momentum = 0.8
    decay_rate = 2e-6
    return lrate


early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=0,
    mode='auto')

bce = tf.keras.losses.BinaryCrossentropy()

model = get_train_model_cosine()
model.compile(loss=bce,
              optimizer=sgd,
              metrics=[accuracy])

history = LossHistory()
lrate = keras.callbacks.LearningRateScheduler(scheduler)
callbacks_ = [history, lrate, early_stopping]
initial_epoch_ = 0
epochs_ = 10

# weights_path = str(models_path / 'sphereface_20_8372.h5')
# model.load_weights(weights_path)

try:
    hist = model.fit(train_generator(train_dataframe, batch_size_),
                     steps_per_epoch=images_per_epoch_ // batch_size_,
                     epochs=epochs_,
                     validation_data=test_generator(
                         test_dataframe, batch_size_),
                     validation_steps=len(test_dataframe) // batch_size_,
                     callbacks=callbacks_,
                     initial_epoch=initial_epoch_,
                     shuffle=True)

    save_path = str(models_path / 'sphereface_20_cosine_new.h5')
except KeyboardInterrupt:
    save_path = str(models_path / 'sphereface_20_cosine_interrupted.h5')

model.save_weights(save_path)
print('Output saved to: "{}./*"'.format(save_path))
