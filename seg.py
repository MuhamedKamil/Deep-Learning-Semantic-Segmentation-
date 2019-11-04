from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os 
import cv2 
from PIL import Image
import time
import IPython.display
import math
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds



#Loading Dataset which consists of image and mask 
dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)

#Intialization Of Parameters 
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH
OUTPUT_CHANNELS = 3


#Encoder Intialization
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

Encoder_Layers = [
    'block_1_expand_relu',   
    'block_3_expand_relu',   
    'block_6_expand_relu',   
    'block_13_expand_relu', 
    'block_16_project',     
]


layers = [base_model.get_layer(name).output for name in Encoder_Layers]
Encoder = tf.keras.Model(inputs=base_model.input, outputs=layers)
Encoder.trainable = False

# Layers Of Decoder
Decoder_Layers = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

#----------------------------------------------------------------------------------------------
#intailization FullModel Unet which consist of two part Encoder for downsampling image and Decoder for upsampling

def Unet_Model(output_channels):

  Last_Layer = tf.keras.layers.Conv2DTranspose(output_channels,3, strides=2, padding='same', activation='softmax')  #64x64 -> 128x128
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  
  OutputOf_encoder = inputs
  skips = Encoder(OutputOf_encoder)
  Input_Of_decoder = skips[-1]
  skips = reversed(skips[:-1])


  for up_sample, skip in zip(Decoder_Layers, skips):
    Input_Of_decoder = up_sample(Input_Of_decoder)
    concat = tf.keras.layers.Concatenate()
    Input_Of_decoder = concat([Input_Of_decoder, skip])

  out_Of_model = Last_Layer(Input_Of_decoder)

  return tf.keras.Model(inputs=inputs, outputs=out_Of_model)

#---------------------------------------------------------------------------------
#Normalize input image between 0 , 1 - class of each pixel in [0 , 1 , 2]
def Normalize_Data(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask
#---------------------------------------------------------------------------------
#Resize Train point which contains input image and true mask to 128 * 128
def resize_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'] , (128,128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'] , (128,128))
    input_image, input_mask = Normalize_Data(input_image, input_mask)
    return input_image, input_mask

#---------------------------------------------------------------------------------
#Resize Test point which contains input image and true mask to 128 * 128
def resize_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
  input_image, input_mask = Normalize_Data(input_image, input_mask)
  return input_image, input_mask

#---------------------------------------------------------------------------------
#Display image - True Mask - Predicted Mask
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

#---------------------------------------------------------------------------------

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
  
#---------------------------------------------------------------------------------

def Model_prediction(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,create_mask(model.predict(sample_image[tf.newaxis, ...]))])


#---------------------------------------------------------------------------------
#mapping function resize_image_train to each data point in training and testing data (image - mask)
train = dataset['train'].map(resize_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(resize_image_test)
#shuffle and batch data
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)
#---------------------------------------------------------------------------------
# visualize sample of data
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])
#---------------------------------------------------------------------------------
EPOCHS = 3
VAL_SUBSPLITS = 5 
VALIDATION_STEPS = info.splits['test'].num_examples
model = Unet_Model(OUTPUT_CHANNELS)
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics= ['accuracy'])
model_history = model.fit(train_dataset, epochs=EPOCHS,steps_per_epoch=STEPS_PER_EPOCH,validation_steps=VALIDATION_STEPS,validation_data=test_dataset,)
Model_prediction(test_dataset, 4)

