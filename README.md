# AgroMed-Model-API

This repository contains a **FastAPI microservice** that predicts crop diseases from images using a **CNN-LSTM model**. You can send an image file or an image URL, and the API will return the predicted crop, disease, and confidence score.

---

## Features

- Predict diseases from uploaded images or image URLs.
- Uses MobileNetV2 + LSTM for predictions.
- Model and class names are stored in `models/`.
- Simple REST API using FastAPI.
- Easy to deploy locally or on cloud.

---

## Folder Structure

```

├── models/
│   ├── crop\_disease\_cnn\_lstm\_model.keras
│   └── class\_names.npy
├── microservice.py
├── requirements.txt
├── README.md
└── data/           # Place your dataset here

```

---

## Dataset

This project uses the **Five Crop Diseases Dataset** from Kaggle:  
[https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset/](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset/)

After downloading, organize it like this:

```

data/
├── Crop1/
│   ├── Disease1/
│   └── Disease2/
├── Crop2/
│   └── ...

````

> Note: `data/` is not included in this repo. Download and place the dataset locally before training.

---

## Installation

1. Clone the repo:

```bash
git clone git@github.com:shubham-parida01/Agromed-Model-API.git
cd Agromed-Model-API
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the API

```bash
uvicorn microservice:app --host 0.0.0.0 --port 8000 --reload
```

* `--reload` restarts the server automatically when code changes.
* API will be available at `http://localhost:8000`.

---

## API Endpoints

### Health Check

**GET /**

Response:

```json
{
    "message": "Crop Disease Prediction API is running!"
}
```

### Predict from Image File

**POST /predict/file**

* Form Data: `file` → upload an image (.jpg, .jpeg, .png)

Response example:

```json
{
  "prediction": {
    "crop": "Tomato",
    "disease": "Leaf Spot"
  },
  "confidence": 0.95
}
```

### Predict from Image URL

**POST /predict/url**

* JSON Body:

```json
{
  "imageUrl": "https://example.com/image.jpg"
}
```

Response is same as above.

---

## Training (Optional)

If you want to retrain the model:

1. Download the dataset from Kaggle.
2. Organize it into `data/` as shown above.
3. Run your training script (e.g., `train.py`) to generate:

* `models/crop_disease_cnn_lstm_model.keras`
* `models/class_names.npy`

---

## Dependencies

* fastapi
* uvicorn
* tensorflow
* opencv-python
* numpy
* requests
* pydantic

---

## Example Usage (Python)

```python
import requests

# Predict from file
files = {"file": open("sample.jpg", "rb")}
response = requests.post("http://localhost:8000/predict/file", files=files)
print(response.json())

# Predict from URL
data = {"imageUrl": "https://example.com/sample.jpg"}
response = requests.post("http://localhost:8000/predict/url", json=data)
print(response.json())
```

---

## License

This project is free to use for learning and hackathon purposes.

```

This can be **copied directly into `README.md`** and pushed to your repo.  

If you want, I can also create a **minimal `requirements.txt`** that matches this microservice so the repo is ready-to-run. Do you want me to do that?
```
