from flask import Flask, render_template, Response, request, jsonify
import cv2
import pickle
import numpy as np
import os
from utils import preprocess_image

app = Flask(__name__)

# Load model
from tensorflow.keras.models import load_model

model = load_model('model/signature_model_v2.h5')
# Camera setup
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.files['image'].read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)
    pred = model.predict([img.flatten()])[0]
    return jsonify({'result': 'Genuine' if pred == 0 else 'Forged'})

    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
