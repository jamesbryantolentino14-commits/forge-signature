from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your trained model (update path if needed)
MODEL_PATH = "signature_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filepath = os.path.join("static", "temp.jpg")
            file.save(filepath)
            # Example: preprocess image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            return f"Prediction: {prediction}"
    return render_template("index.html")

if __name__ == "__main__":
    # Listen on all interfaces, Render sets the port via $PORT
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)