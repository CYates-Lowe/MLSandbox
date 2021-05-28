# -*- coding: utf-8 -*-
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Input, Dense, Reshape, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.activations import selu, tanh, sigmoid 
from tensorflow.keras.initializers import lecun_normal, he_normal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

cross_entropy = BinaryCrossentropy(from_logits=True)

def Generator(input_shape=(256,)):
    inputs = Input(shape=input_shape, name='InputsLayer')
    
    X = Dense(7*7*256, use_bias=False)(inputs)
    X = Reshape((7,7,256))(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    X = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(1,1), padding='same', use_bias=False)(X) 
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    X = Conv2DTranspose(filters=64, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    outputs = Conv2DTranspose(filters=1, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation=tanh)(X)

    return Model(inputs=inputs, outputs=outputs, name='GeneratorModel')

def GeneratorLoss(y):
    return cross_entropy(tf.ones_like(y), y)

def Discriminator(input_shape=(28,28,1)):
    inputs = Input(shape=input_shape, name='InputsLayer')
    
    X = Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation=selu, kernel_initializer=lecun_normal)(inputs)
    X = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding='same', use_bias=False, activation=selu, kernel_initializer=lecun_normal)(X)
    
    X = Flatten()(X)
    
    outputs = Dense(1, activation=sigmoid)(X)
    
    return Model(inputs=inputs, outputs=outputs, name='DiscriminatorModel')

def DiscriminatorLoss(x, y):
    real_loss = BinaryCrossentropy()(tf.ones_like(x), x)
    fake_loss = BinaryCrossentropy()(tf.zeros_like(y), y)
    loss_sum = real_loss+fake_loss
    return loss_sum

def GenSaveImgs(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()


@tf.function
def TrainStep(images):
    noise = tf.random.normal([256, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = GeneratorLoss(fake_output)
        disc_loss = DiscriminatorLoss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
        opt_a.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        opt_b.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
    
        for image_batch in dataset:
            TrainStep(image_batch)
        
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        GenSaveImgs(generator, epoch + 1, seed)
            
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        
        # Generate after the final epoch
        display.clear_output(wait=True)
        GenSaveImgs(generator, epochs, seed)

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = Generator()
noise = tf.random.normal([1, 256])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0,:,:,0], cmap='gray')

discriminator = Discriminator()
decision = discriminator(generated_image)
print(decision)

opt_a = Adam(learning_rate=1e-4)
opt_b = Adam(learning_rate=1e-4)

epochs = 50
noise_dim = 256
num_to_gen = 8

seed = tf.random.normal([num_to_gen, noise_dim])

train(train_dataset, epochs)
