import cv2
import numpy as np

# ==============================
# Preprocess Image for Model
# ==============================
def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    return img


# ==============================
# Check if Image Looks Like Signature
# ==============================
def is_signature_like(image):
    mean_intensity = np.mean(image)

    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # Too dark
    if mean_intensity < 40:
        return False

    # Too complex (face/object)
    if edge_density > 40:
        return False

    return True