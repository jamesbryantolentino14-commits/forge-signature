import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

def load_images(folder):
    X, y = [], []
    for label, subfolder in enumerate(['genuine', 'forged']):
        path = os.path.join(folder, subfolder)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128, 64))
            img = img / 255.0  # Normalize
            X.append(img.flatten())
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_images("dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
with open('model/signature_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'model/signature_model.pkl'")