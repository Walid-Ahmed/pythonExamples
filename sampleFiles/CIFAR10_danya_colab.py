#!/usr/bin/env python
# coding: utf-8

# # CIFAR-10

# In[1]:

from __future__ import print_function
from matrix_square_root_power import *
from shampoo_optimizer import *
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, Sequential
from keras.datasets import cifar10
from keras import backend
from keras.utils import np_utils
import numpy
from keras.datasets import cifar10
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# In[ ]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# In[ ]:


train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images / 255.0
test_images = test_images / 255.0


# In[ ]:


def loss(model, x, y, loss_object):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, loss_object)
        return loss_value, tape.gradient(loss_value, model.trainable_weights)


# In[ ]:


train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)


# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = ShampooOptimizer(learning_rate=0.01)

shampoo_train_loss_results = []
shampoo_train_accuracy_results = []

epochs = 50
batch_size = 128

for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 128
    for i in range(train_images.shape[0] // batch_size):
        x = train_images[i*batch_size:i*batch_size+batch_size]
        y = train_labels[i*batch_size:i*batch_size+batch_size]
        # Optimize the model
        loss_value, grads = grad(model, x, y, loss_object)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        epoch_accuracy(y, model(x))

        # End epoch
    shampoo_train_loss_results.append(epoch_loss_avg.result())
    shampoo_train_accuracy_results.append(epoch_accuracy.result())

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))


# In[ ]:


get_ipython().system('pip install tensorflow==2.1.0-rc1')


# In[ ]:


from keras.datasets import mnist
#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.summary()


# In[ ]:


import tensorflow as tf
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)


# In[ ]:


def loss(model, x, y, loss_object):
    y_ = model(x)

    return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, loss_object)
        return loss_value, tape.gradient(loss_value, model.trainable_weights)


# In[ ]:


import tensorflow as tf
model_shampoo_use_iterative_root = Sequential()
#add model layers
model_shampoo_use_iterative_root.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28,28,1)))
model_shampoo_use_iterative_root.add(Conv2D(32, kernel_size=3, activation="relu"))
model_shampoo_use_iterative_root.add(Flatten())
model_shampoo_use_iterative_root.add(Dense(10, activation="softmax"))
model_shampoo_use_iterative_root.summary()
shampoo_use_iterative_root_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
shampoo_use_iterative_root_optimizer = ShampooOptimizer(learning_rate=5*1e-5)

shampoo_use_iterative_root_train_loss_results = []
shampoo_use_iterative_root_train_accuracy_results = []

epochs = 20
batch_size = 128

for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 128
    for i in range(X_train.shape[0] // batch_size):
        x = X_train[i*batch_size:i*batch_size+batch_size]
        y = y_train[i*batch_size:i*batch_size+batch_size]
        # Optimize the model
        loss_value, grads = grad(model_shampoo_use_iterative_root, x, y, shampoo_use_iterative_root_loss_object)
        shampoo_use_iterative_root_optimizer.apply_gradients(zip(grads, model_shampoo_use_iterative_root.trainable_weights))

        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        epoch_accuracy(y, model_shampoo_use_iterative_root(x))

        # End epoch
    shampoo_use_iterative_root_train_loss_results.append(epoch_loss_avg.result())
    shampoo_use_iterative_root_train_accuracy_results.append(epoch_accuracy.result())

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

