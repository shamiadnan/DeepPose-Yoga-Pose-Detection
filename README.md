# DeepPose: AI-Powered Yoga Pose Recognition and Correction

DeepPose is a real-time yoga posture recognition and correction system built using **PyTorch, MediaPipe, and OpenCV**.  
It acts as a virtual yoga instructor, analyzing poses via webcam and providing instant corrective feedback to ensure safe and effective practice.

---

## 📌 Features
- Real-time yoga pose recognition using **ResNet-18** (transfer learning).
- Landmark detection with **MediaPipe Pose** (33 skeletal points).
- Rule-based feedback engine for corrective guidance (e.g., "Bend knee more").
- Overlay visualization: skeleton, pose label, confidence score, feedback text.
- Supports 5 yoga poses: **Downdog, Goddess, Plank, Tree, Warrior II**.
- Lightweight, runs locally on CPU/GPU with webcam input.

---

## 📂 Project Structure
DeepPose/
│── main.py                                # Entry point for real-time detection
│── config.py                            # Configuration (paths, thresholds, device)
│── requirements.txt              # Dependencies
│── checkpoints/           # Trained model weights (.pth file)
│── logs/                  # Training logs
│── docs/                  # Diagrams, training curves, screenshots
│── examples/              # Sample input/output images


---


## 📊 Results & Analysis
- **Training Accuracy:** ~90% validation accuracy across 25 epochs.
- **Training Curves:**
  - Loss steadily decreased for both training and validation.
  - Accuracy stabilized around 0.9, showing strong generalization.
- **Pose Recognition:**
  - High accuracy: Plank, Tree, Downdog.
  - Lower confidence: Warrior II, Goddess (due to pose similarity and dataset imbalance).
- **Real-Time Demo:** Achieved >25 FPS on standard laptop webcam.

<img width="1500" height="600" alt="training_curves" src="https://github.com/user-attachments/assets/c96cefb9-7b4d-4054-a066-6ef613f2b70d" />
<img width="873" height="636" alt="webcam demo" src="https://github.com/user-attachments/assets/9eac7094-5180-4356-a95b-abf6c9ce5319" />

---

## 🚀 Future Enhancements
- Expand dataset to cover more yoga poses and should be train the model by videos for better accuracy.
- Mobile deployment (TensorFlow Lite / ONNX).
- Voice-based feedback (Text-to-Speech).
- Sequence modeling with LSTM/Transformers for dynamic yoga flows.
- Gamification and AR guidance for interactive practice.
- Cloud integration for personalized progress tracking.

---

## 📚 References
- PyTorch Documentation
- MediaPipe Pose
- Toshev, A., & Szegedy, C. (2014). DeepPose: Human Pose Estimation via Deep Neural Networks. CVPR.
- Kaggle Yoga Dataset

---

## 👨‍💻Author
- **Adnan Shami**
- Guided by **Prof. Ramkrishna Pal**
