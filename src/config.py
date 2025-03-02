import os

# Paths
RAW_DATASET_PATH = "data/raw/"
PROCESSED_DATASET_PATH = "data/processed/"
TRAIN_FOLDER = os.path.join(PROCESSED_DATASET_PATH, "train")
TEST_FOLDER = os.path.join(PROCESSED_DATASET_PATH, "test")

# Image processing
TARGET_SIZE = (224, 224)  # Image size for CNN input
AUGMENTATION_MULTIPLIER = 10  # Create 10x more images

# Training parameters
EPOCHS = 10
BATCH_SIZE = 8
