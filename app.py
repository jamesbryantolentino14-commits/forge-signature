from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pickle
import os
from utils import preprocess_image, is_signature_like

app = Flask(__name__)

# ==============================
# Load Trained Model
# ==============================
model_path = os.path.join("model", "signature_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)


# ==============================
# Signature Validation Function
# ==============================
def is_signature_like(image):
    """
    Prevent faces or random photos from being detected as genuine.
    """

    mean_intensity = np.mean(image)

    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # Too dark
    if mean_intensity < 40:
        return False

    # Too complex (like face or object)
    if edge_density > 40:
        return False

    return True


# ==============================
# Routes
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"result": "No image received", "status": "unreliable"})

    file = request.files["image"]
    data = file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({"result": "Invalid image", "status": "unreliable"})

    from utils import preprocess_image, is_signature_like

    # Check if looks like a signature
    if not is_signature_like(img):
        return jsonify({"result": "Unreliable Image (Not a Signature)", "status": "unreliable"})

    # Preprocess for model
    img = preprocess_image(img)

    # Predict
    pred = model.predict([img.flatten()])[0]

    if pred == 0:
        return jsonify({"result": "Genuine Signature", "status": "genuine"})
    else:
        return jsonify({"result": "Forged Signature", "status": "forged"})


# ==============================
# Run App (Render Compatible)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)