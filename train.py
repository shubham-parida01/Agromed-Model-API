# training.py

import os
from sklearn.model_selection import train_test_split

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
    return X_train, y_train, X_temp, y_temp

print("Dataset loading function added")
