import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# settings
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
BASE_DIR = "data"
MODEL_DIR = "models"

# make sure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

# load dataset from folders
def create_dataset(base_dir):
    X, y, class_names = [], [], []
    label_map = {}
    label_idx = 0

    for crop in os.listdir(base_dir):
        crop_path = os.path.join(base_dir, crop)
        if not os.path.isdir(crop_path):
            continue
        for disease in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease)
            if not os.path.isdir(disease_path):
                continue
            class_name = f"{crop}_{disease}"
            class_names.append(class_name)
            label_map[class_name] = label_idx
            label_idx += 1

            for img_file in os.listdir(disease_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(disease_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                X.append(img)
                y.append(label_map[class_name])

    X = np.array(X)
    y = np.array(y)
    return X, y, class_names

# make model
def build_cnn_lstm_model(input_shape, num_classes):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights="imagenet")
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Reshape((-1, base_model.output_shape[-1])),
        layers.LSTM(128),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# train
def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS):
    datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest"
    )

    model_path = os.path.join(MODEL_DIR, "crop_disease_cnn_lstm_model.keras")
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stop, checkpoint]
    )
    return history

# test
def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")
    return acc

# single image helper
def preprocess_image(img_path, img_size=IMG_SIZE):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_single_image(model, img_path, class_names):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    top_idx = np.argmax(preds[0])
    confidence = float(preds[0][top_idx])
    return class_names[top_idx], confidence

# save classes
def save_class_names(class_names, filename="class_names.npy"):
    file_path = os.path.join(MODEL_DIR, filename)
    np.save(file_path, class_names)
    print(f"saved classes to {file_path}")

# run pipeline
if __name__ == "__main__":
    X, y, class_names = create_dataset(BASE_DIR)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model = build_cnn_lstm_model((IMG_SIZE, IMG_SIZE, 3), len(class_names))
    train_model(model, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test)
    save_class_names(class_names)
