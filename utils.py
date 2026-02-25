import cv2
import numpy as np

def preprocess_image(img):
    img = cv2.resize(img, (128, 64))
    img = img / 255.0
    return img

def is_signature_like(image):
    mean_intensity = np.mean(image)
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    if mean_intensity < 20:  # too dark
        return False
    if edge_density > 150:    # too complex (like a face)
        return False

    return True