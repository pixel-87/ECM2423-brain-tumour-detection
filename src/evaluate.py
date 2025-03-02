from tensorflow.keras.models import load_model
from config import TEST_FOLDER
import numpy as np
import os
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for category in ["no", "yes"]:
        label = 0 if category == "no" else 1
        category_path = os.path.join(folder, category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Ensure the image is resized to 224x224
                images.append(img)
                labels.append(label)
    images = np.array(images)
    images = images.reshape(images.shape[0], 224, 224, 1)  # Reshape to (num_samples, 224, 224, 1)
    return images, np.array(labels)

# Load preprocessed test data
X_test, y_test = load_images_from_folder(TEST_FOLDER)

# Normalize the images
X_test = X_test / 255.0

# Load the trained model
model = load_model('models/trained_model.keras')  # Update with the actual path to your trained model

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
logging.info(f"Test Accuracy: {test_acc * 100:.2f}%")
