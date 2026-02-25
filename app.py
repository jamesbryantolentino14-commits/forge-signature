from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import pickle
import os
from utils import preprocess_image

app = Flask(__name__)

# ==============================
# Load Trained Model
# ==============================
model_path = os.path.join("model", "signature_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ==============================
# Camera Setup (Local Only)
# ==============================
camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )


# ==============================
# Signature Validation Function
# ==============================
def is_signature_like(image):
    """
    Prevent faces or random photos from being detected as genuine.
    """

    # Mean brightness
    mean_intensity = np.mean(image)

    # Edge detection
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # If too dark or too complex → unreliable
    if mean_intensity < 40:
        return False

    if edge_density > 40:
        return False

    return True


# ==============================
# Routes
# ==============================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    data = file.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    # Validate image first
    if not is_signature_like(img):
        return jsonify({
            "result": "Unreliable Image (Not a Signature)",
            "status": "unreliable"
        })

    # Preprocess
    img = preprocess_image(img)

    # Prediction
    pred = model.predict([img.flatten()])[0]

    if pred == 0:
        return jsonify({
            "result": "Genuine Signature",
            "status": "genuine"
        })
    else:
        return jsonify({
            "result": "Forged Signature",
            "status": "forged"
        })


# ==============================
# Run App (Render Compatible)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)