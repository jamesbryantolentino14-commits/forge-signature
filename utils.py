import cv2
import numpy as np

def is_signature_like(image):
    """
    Simple validation:
    Signature should have white background and thin dark strokes.
    If too dark or too complex (like a face), mark as unreliable.
    """

    # Check brightness
    mean_intensity = np.mean(image)

    # Check edge density
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # Conditions for unreliable image
    if mean_intensity < 50:      # too dark
        return False
    if edge_density > 40:       # too complex (like face)
        return False

    return True