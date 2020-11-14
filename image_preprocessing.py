#IMAGE PREPROCESSING 

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import model_from_json

import numpy

import os

import matplotlib.pyplot as plt

#Remove corrupted files

num_skipped = 0

for folder in ("test", "train"):
    for folder_name in ("NORMAL", "PNEUMONIA"):

        folder_path = os.path.join("Pediatric Chest X-ray Pneumonia", folder, folder_name)

        for fname in os.listdir(folder_path):

            fpath = os.path.join(folder_path, fname)

            try:

                fobj = open(fpath, "rb")

                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)

            finally:

                fobj.close()

    

            if not is_jfif:

                num_skipped += 1

                # Delete corrupted image

                os.remove(fpath)

print("Deleted %d images" % num_skipped)

#Image Preprocessing

image_size = (256,256)
batch_size = 1

!pwd

train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "Pediatric Chest X-ray Pneumonia/train",

    color_mode = "rgb",

    image_size=image_size

)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    "Pediatric Chest X-ray Pneumonia/test",

    color_mode = "rgb",

    image_size=image_size

)

##########. TO DO: DISPLAY IMAGES #####################

# # check first 9 images in training set: 1 = dog, 0 = cat

# plt.figure(figsize=(10, 10))

# for images, labels in train_ds.take(1):

#     for i in range(9):

#         ax = plt.subplot(3, 3, i + 1)

#         plt.imshow(images[i], aspect = "auto")

#         plt.title(int(labels[i]))

#         plt.axis("off")

 

# data augmentation - rotate some images 

data_augmentation = keras.Sequential(

    [

        layers.experimental.preprocessing.RandomFlip("horizontal"),

        layers.experimental.preprocessing.RandomRotation(0.1),

    ]

)

 
############# FIX DATA AUGMENTATION DISPLAY ###################
# # first image after augmentations

# plt.figure(figsize=(10, 10))

# for images, _ in train_ds.take(1):

#     for i in range(9):

#         augmented_images = data_augmentation(images)

#         ax = plt.subplot(3, 3, i + 1)

#         plt.imshow(augmented_images[0])

#         plt.axis("off")

 

# # buffer the data to improve IO

train_ds = train_ds.prefetch(buffer_size=32)

val_ds = val_ds.prefetch(buffer_size=32)