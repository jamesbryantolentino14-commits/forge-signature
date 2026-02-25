from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("signature_model_v2.h5")

# Ensure the static folder exists
UPLOAD_FOLDER = "static"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))
    return img

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, "temp.jpg")
            file.save(filepath)
            img = preprocess_image(filepath)
            prediction = model.predict(img)[0][0]

            if prediction > 0.5:
                result = "Forged Signature"
            else:
                result = "Genuine Signature"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)