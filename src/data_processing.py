import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import RAW_DATASET_PATH, PROCESSED_DATASET_PATH, TRAIN_FOLDER, TEST_FOLDER, TARGET_SIZE, AUGMENTATION_MULTIPLIER
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define data augmentation strategy
datagen = ImageDataGenerator(
    # rotation_range=15,        # Rotate images up to 15 degrees
    height_shift_range=0.1,   # Shift height by 10%
    zoom_range=0.1,           # Small zoom
    horizontal_flip=True,     # Flip images horizontally
)

# Ensure processed dataset directories exist
for folder in [TRAIN_FOLDER, TEST_FOLDER]:
    for category in ["yes", "no"]:
        category_path = os.path.join(folder, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

def load_and_process_images(input_folder, target_size):
    """
    Loads images from 'yes' and 'no' folders, resizes them, converts to grayscale,
    and normalizes pixel values.
    """
    images = []
    labels = []
    total_images_found = 0
    total_images_loaded = 0
    
    for category in ["no", "yes"]:
        label = 0 if category == "no" else 1
        input_category_path = os.path.join(input_folder, category)

        for filename in os.listdir(input_category_path):
            total_images_found += 1  # Count all found files
            img_path = os.path.join(input_category_path, filename)

            # Check if the file is a supported image format
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

                if img is None:
                    logging.warning(f"Could not load {img_path}. Skipping.")
                    continue  # Skip unreadable images

                total_images_loaded += 1
                
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  # Resize

                # Normalize pixel values to range [0,1]
                img = img / 255.0
                
                images.append(img)
                labels.append(label)

    logging.info(f"Found {total_images_found} images in dataset.")
    logging.info(f"Successfully loaded {total_images_loaded} images.")

    images = np.array(images)
    images = images.reshape(images.shape[0], target_size[0], target_size[1], 1)  # Reshape to (num_samples, 224, 224, 1)
    return images, np.array(labels)

def save_images(X, y, folder):
    """
    Saves images into respective folders (yes/no) inside the train or test directory.
    """
    for i in range(len(X)):
        category = "yes" if y[i] == 1 else "no"
        save_path = os.path.join(folder, category)
        
        img = (X[i] * 255).astype(np.uint8)  # Convert back to 0-255 range
        filename = f"img_{i}.png"
        cv2.imwrite(os.path.join(save_path, filename), img)

def augment_images(X_train, y_train, output_folder, multiplier):
    """
    Applies data augmentation only to the training dataset and saves the augmented images.
    """
    for i in range(len(X_train)):
        img = np.expand_dims(X_train[i], axis=0)  # Add batch dimension

        j = 0
        for batch in datagen.flow(
            img,
            batch_size=1,
            save_to_dir=os.path.join(output_folder, "yes" if y_train[i] == 1 else "no"),
            save_prefix=f"aug_{i}",
            save_format="png"
        ):
            j += 1
            if j >= multiplier:
                break  # Stop after generating 'multiplier' augmented images

# Load and preprocess images
X, y = load_and_process_images(RAW_DATASET_PATH, TARGET_SIZE)

# Split into train (80%) and test (20%) before augmentation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
logging.info(f"Total images before split: {len(X)}")
logging.info(f"Train set (before augmentation): {len(X_train)}")
logging.info(f"Test set: {len(X_test)}")

# Save original train and test images separately
save_images(X_train, y_train, TRAIN_FOLDER)
save_images(X_test, y_test, TEST_FOLDER)

# Augment only training images
augment_images(X_train, y_train, TRAIN_FOLDER, AUGMENTATION_MULTIPLIER)

logging.info("Dataset successfully processed! Train and test sets created with correct augmentation.")
