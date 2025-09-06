import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
MODEL_DIR = "models"

def predict_image(img_path):
    # load model and classes
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "crop_disease_cnn_lstm_model.keras"))
    class_names = np.load(os.path.join(MODEL_DIR, "class_names.npy"), allow_pickle=True).tolist()

    # read and prepare image
    img = cv2.imread(img_path)
    if img is None:
        return "Invalid image", 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # predict
    preds = model.predict(img)
    idx = np.argmax(preds[0])
    return class_names[idx], float(preds[0][idx])

if __name__ == "__main__":
    label, conf = predict_image("test_samples/sample.jpg")  # change path as needed
    print(f"Prediction: {label} ({conf:.2f})")
