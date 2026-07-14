"""
DeepPose 2.0 Configuration
==========================
Central configuration for the AI-powered Yoga Pose Recognition and Correction System.
All hyperparameters, paths, and constants are defined here.
"""

import os
import torch

# ========================
# Project Paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories if they don't exist
for dir_path in [CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ========================
# Dataset Configuration
# ========================
CLASSES = [
    "Bridge_Pose",
    "Chair_Pose",
    "Child_Pose",
    "Cobra_Pose",
    "Downdog_Pose",
    "Goddess_Pose",
    "Plank_Pose",
    "Tree_Pose",
    "Triangle_Pose",
    "Warrior1_Pose",
    "Warrior2_Pose",
]
NUM_CLASSES = len(CLASSES)  # 11

# ========================
# Training Hyperparameters
# ========================
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 5  # Early stopping patience
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# ========================
# Model Configuration
# ========================
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
PRETRAINED = True
FREEZE_BACKBONE = True

# ========================
# Device Configuration
# ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# ImageNet Normalization
# ========================
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ========================
# Real-Time Inference
# ========================
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 10  # Rolling window for frame smoothing
HOLD_FRAMES_REQUIRED = 10  # Frames to confirm pose hold (~0.33s at 30fps)

# ========================
# Scoring & Feedback
# ========================
DEVIATION_THRESHOLD = 15.0  # Degrees - minimum deviation to trigger feedback
MAX_DEVIATION = 45.0  # Degrees - maximum deviation for scoring normalization

# ========================
# Voice Configuration
# ========================
VOICE_RATE = 150  # Words per minute for pyttsx3

# ========================
# MediaPipe Configuration
# ========================
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ========================
# Streamlit Configuration
# ========================
APP_TITLE = "DeepPose 2.0"
APP_DESCRIPTION = "AI-Powered Yoga Pose Recognition & Correction System"
MAX_UPLOAD_SIZE_MB = 10
