# training.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

def create_dataset(dataset_dir, test_size=0.2, val_size=0.1, seed=SEED):
    classes = sorted(os.listdir(dataset_dir))
    images, labels = [], []
    for label, crop in enumerate(classes):
        crop_dir = os.path.join(dataset_dir, crop)
        for img_file in os.listdir(crop_dir):
            images.append(os.path.join(crop_dir, img_file))
            labels.append(label)

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=test_size+val_size, stratify=labels, random_state=seed
    )

    val_rel_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_rel_size, stratify=y_temp, random_state=seed
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0/255)


