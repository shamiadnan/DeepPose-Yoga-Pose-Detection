# 🧘 DeepPose 2.0 — AI-Powered Yoga Pose Recognition & Correction

An intelligent real-time yoga coaching system that recognizes 11 yoga poses, scores your posture quality, and provides corrective voice feedback — all through a web dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **11 Yoga Pose Recognition** — Bridge, Chair, Child, Cobra, Downdog, Goddess, Plank, Tree, Triangle, Warrior I, Warrior II
- **ResNet18 Transfer Learning** — Pretrained on ImageNet, fine-tuned for yoga classification
- **Real-Time Skeleton Tracking** — MediaPipe Pose with 33 body landmarks
- **Posture Scoring (0-100)** — Biomechanical angle analysis against ideal poses
- **Corrective Feedback** — Rule-based, anatomically aware correction instructions
- **Voice Guidance** — Non-blocking pyttsx3 text-to-speech
- **Streamlit Dashboard** — 4-page web interface (Home, Upload, Camera, Metrics)
- **State Machine** — IDLE → TRANSITIONING → HOLDING_POSE for stable detection

## 🏗️ Architecture

```
Camera/Image → ResNet18 → Pose Classification (11 classes)
             → MediaPipe → 33 Landmarks → Skeleton Overlay
             → Angle Calculator → 7 Joint Angles
             → Posture Scorer → Score (0-100)
             → Feedback Engine → Corrective Instructions
             → Voice Engine → Spoken Guidance
```

## 📂 Project Structure

```
DeepPose-Yoga-Pose-Detection/
├── config.py                  # Central configuration (11 classes, hyperparams)
├── models/
│   ├── train.py              # DatasetManager + YogaTrainer (ResNet18)
│   ├── evaluate.py           # Metrics, confusion matrix, learning curves
│   ├── inference.py          # PoseInferenceEngine (GPU→CPU fallback)
│   └── session.py            # State machine (IDLE/TRANSITIONING/HOLDING)
├── scoring/
│   ├── angle_utils.py        # AngleCalculator (7 joints, dot product)
│   ├── posture_score.py      # PostureScorer (ideal angles DB, 0-100)
│   └── feedback_engine.py    # FeedbackEngine (rule-based corrections)
├── voice/
│   └── voice_feedback.py     # VoiceFeedback (non-blocking pyttsx3)
├── mediapipe_utils/
│   └── pose_detector.py      # PoseDetector (landmarks, skeleton, features)
├── streamlit_app/
│   └── app.py                # Multi-page Streamlit dashboard
├── checkpoints/              # Saved model weights (best_model.pth)
├── outputs/                  # Generated plots (confusion matrix, curves)
├── dataset/                  # Training images (11 class subdirectories)
├── requirements.txt          # Python dependencies
├── run.py                    # Main entry point (train/evaluate/app)
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your yoga pose images in the `dataset/` directory:
```
dataset/
├── Bridge_Pose/
├── Chair_Pose/
├── Child_Pose/
├── Cobra_Pose/
├── Downdog_Pose/
├── Goddess_Pose/
├── Plank_Pose/
├── Tree_Pose/
├── Triangle_Pose/
├── Warrior1_Pose/
└── Warrior2_Pose/
```

### 3. Train the Model

```bash
python -c "
from models.train import DatasetManager, YogaTrainer
from models.evaluate import evaluate_model, plot_confusion_matrix, plot_training_curves

# Load and split dataset
manager = DatasetManager('dataset/', img_size=224, batch_size=32)
train_loader, val_loader, test_loader = manager.load_and_split()
class_weights = manager.get_class_weights()

# Train
trainer = YogaTrainer(num_classes=11, patience=5)
history = trainer.train(train_loader, val_loader, epochs=50, class_weights=class_weights)
trainer.save_checkpoint()

# Evaluate
metrics = evaluate_model(trainer.model, test_loader, manager.get_class_names())
plot_confusion_matrix(metrics['confusion_matrix'], manager.get_class_names())
plot_training_curves(history)
"
```

### 4. Launch Dashboard

```bash
streamlit run streamlit_app/app.py
```

## 📊 Supported Yoga Poses

| # | Pose | Key Joints |
|---|------|-----------|
| 1 | Bridge Pose | Knee 90°, Hip lifted |
| 2 | Chair Pose | Knee 100°, Hip 100° |
| 3 | Child Pose | Knee folded, Hip deep |
| 4 | Cobra Pose | Legs straight, Elbows bent |
| 5 | Downdog Pose | Legs straight, Hips high |
| 6 | Goddess Pose | All joints 90° |
| 7 | Plank Pose | Body straight line (180°) |
| 8 | Tree Pose | Standing leg straight, Other bent |
| 9 | Triangle Pose | Legs straight, Hips 90° |
| 10 | Warrior I | Front knee 90°, Back leg straight |
| 11 | Warrior II | Front knee 90°, Arms level |

## 🧪 How Scoring Works

1. **Angle Calculation** — 7 joint angles measured using vector dot product
2. **Deviation** — Absolute difference from ideal angle for each joint
3. **Weighted Score** — Penalty normalized to [0, 100] using pose-specific joint weights
4. **Feedback** — Corrections generated for deviations > 15° (sorted by severity)

## 🔧 Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `NUM_CLASSES` | 11 | Number of yoga poses |
| `IMG_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer LR |
| `PATIENCE` | 5 | Early stopping epochs |
| `CONFIDENCE_THRESHOLD` | 0.5 | Minimum prediction confidence |
| `DEVIATION_THRESHOLD` | 15.0° | Feedback trigger threshold |
| `HOLD_FRAMES_REQUIRED` | 10 | Frames to confirm pose hold |

## 📈 Performance Targets

| Component | Target |
|-----------|--------|
| ResNet18 (GPU) | < 20ms |
| ResNet18 (CPU) | < 50ms |
| MediaPipe | < 15ms |
| Angle Calculation | < 1ms |
| Posture Scoring | < 1ms |
| Full Pipeline (30fps) | < 33ms |

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) — Google's pose estimation
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Streamlit](https://streamlit.io/) — Web dashboard framework
- ImageNet — Pretrained weights for transfer learning
