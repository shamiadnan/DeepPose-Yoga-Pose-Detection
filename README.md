##DeepPose – Yoga Pose Recognition & Real-Time
DeepPose is an AI-powered yoga pose recognition system that uses deep learning, sequence modeling, and pose estimation to classify yoga poses and provide real-time correction feedback.
The system is built for academic research, fitness applications, and real-time digital coaching.

##⭐ Project Features
Yoga Pose Recognition from videos or webcam

Pose keypoint extraction using MoveNet / Mediapipe

Keypoint sequence generation (30-frame sliding window)

Conv1D + LSTM model for temporal yoga pose classification

Real-time pose detection on webcam

Real-time correction feedback using angle deviations

Complete dataset pipeline

Google Colab training notebook included

##🧠 Tech Stack
Python 3.10

PyTorch

TensorFlow Hub (MoveNet)

Mediapipe

OpenCV

NumPy / Pandas

Google Colab (GPU training)

VS Code (Realtime inference)

📂 Repository Structure
bash
Copy code
DeepPose/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── DeepPose_Training_Notebook.ipynb
│
├── src/
│   ├── extract_sequences.py
│   ├── train_model.py
│   ├── realtime_demo.py
│   ├── model.py
│   ├── utils.py
│   └── angle_templates.py
│
├── data/
│   ├── raw_videos/                 # Your training videos go here
│   ├── processed_sequences/
│   ├── sequences/
│   └── templates/
│
├── models/
│   ├── deeppose_seq_best.pth       # Trained sequence model
│   └── template_angles.npy         # Correction template
│
└── results/
    ├── confusion_matrix.png
    └── training_logs/
📥 Dataset Preparation
Create folders for each pose:

bash
Copy code
data/raw_videos/
  tree/
  warrior/
  mountain/
  triangle/
  downward_dog/
Add 10–50 videos per pose (5–10 seconds each).
Alternatively, use images (code supports both).

🧪 Training Pipeline (Google Colab)
The notebook inside /notebooks performs:

Install dependencies

Upload dataset

Extract pose keypoints using MoveNet

Build fixed-length keypoint sequences

Train ConvLSTM model

Evaluate + save best model

Generate correction angle templates

The final trained model appears in /models.

🎥 Real-Time Yoga Detection (Local VS Code)
Run:

bash
Copy code
python src/realtime_demo.py
Features:

Webcam-based pose recognition

Realtime visual overlay

Real-time correction messages

Works on CPU or GPU
``
📊 Results
High accuracy for 5–6 yoga classes

Stable real-time pose recognition

Correction system based on angle deviation

Confusion matrix and logs included
``
🚀 Future Work
Add more yoga poses

Use Vision Transformers for pose classification

Deploy as a web app (Gradio / Streamlit)

Mobile version with TensorFlow Lite

👤 Author
Adnan Shami
