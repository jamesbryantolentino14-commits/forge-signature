# app.py
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("signature_model_v2.h5")

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = "static/temp.jpg"
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)[0][0]

    # Correct label logic
    confidence = round(float(prediction)*100, 2)  # 0-100%
    if prediction > 0.5:
        result = "Genuine Signature"
    else:
        result = "Forged Signature"

    # Send to template
    return render_template("index.html", prediction=result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)