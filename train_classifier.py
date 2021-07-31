import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Nadam
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_path = './Dataset/Train/'
val_path = './Dataset/Validation/'
# Hyperparameters
lr = 1e-3
batch_size = 16
img_height, img_width = 331, 331
classes = 2
epochs=35
#Data Generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30,
    shear_range=0.5,
    zoom_range=.7,
    channel_shift_range=0.3,
    cval=0.5,
    vertical_flip=True,
    fill_mode='nearest',validation_split=0.2)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator =train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_path,target_size=(img_height,img_width),batch_size=batch_size,class_mode='categorical')
#callbacks
my_callbacks = [tf.keras.callbacks.ModelCheckpoint('./model/mask_model.h5',monitor='val_loss',verbose=1,save_best_only=True),
               tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1,patience=7)]

#Model
model_res = ResNet50(include_top=False, weights='imagenet',
                    input_shape=(img_height, img_width, 3))
x = model_res.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(classes)(x)
model = Model(inputs=model_res.input, outputs=x)

for layer in model_res.layers:
    layer.trainable=False
    
model.compile(optimizer=Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.__len__(),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_generator.__len__(),
                    callbacks=my_callbacks)