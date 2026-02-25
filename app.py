from flask import Flask, render_template, request
import numpy as np
import cv2
import os
import tensorflow as tf

app = Flask(__name__)

# Load model using environment variable
MODEL_PATH = os.getenv("MODEL_PATH", "signature_model_v2.h5")
model = tf.keras.models.load_model(MODEL_PATH)

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
    result = "Forged Signature" if prediction > 0.5 else "Genuine Signature"
    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))