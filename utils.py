import cv2
import numpy as np

def preprocess_image(img):
    img = cv2.resize(img, (128, 64))
    img = img / 255.0
    return img

def is_signature_like(img):
    mean_intensity = np.mean(img)
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])

    if mean_intensity < 80:    # too dark
        return False
    if edge_density > 0.15:      # too complex
        return False
    return True