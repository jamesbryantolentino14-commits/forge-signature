import cv2
import numpy as np

def preprocess_image(img):
    img = cv2.resize(img, (128, 64))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img