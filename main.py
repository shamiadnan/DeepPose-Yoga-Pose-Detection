#Import required libraries
import cv2, torch, torch.nn as nn, numpy as np
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from collections import deque
import mediapipe as mp
from config import *

# Load trained checkpoints and retsore ResNet-18 model architecture
ckpt = torch.load(CKPT_DIR / "resnet18_yoga_best.pth", map_location=DEVICE)
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()
classes = ckpt["classes"]

# Preprocessing transforms
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

#Helper function: preprocess OpenCV frame for ResNet
def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    from PIL import Image
    img = Image.fromarray(rgb)
    return val_tf(img).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(x):
    out = model(x)
    probs = torch.softmax(out, dim=1)
    conf, idx = torch.max(probs, dim=1)
    return float(conf), classes[idx.item()]

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def feedback(label, lm, w, h):
    fb = []
    if lm is None: return fb
    def pt(l): return (int(l.x * w), int(l.y * h))
    hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    sho = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    rsho = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    ang_knee = angle(pt(hip), pt(knee), pt(ankle))
    shoulder_line = abs(pt(sho)[1] - pt(rsho)[1])
    if "warrior" in label.lower():
        if ang_knee > 110: fb.append("Bend knee more.")
        if shoulder_line > 20: fb.append("Level shoulders.")
    if "goddess" in label.lower():
        if ang_knee < 90: fb.append("Lower squat.")
    if "plank" in label.lower():
        fb.append("Keep hips level.")
    if "tree" in label.lower():
        fb.append("Engage core for balance.")
    if "downdog" in label.lower():
        fb.append("Lift hips higher.")
    return fb

# Webcam loop
cap = cv2.VideoCapture(2)
buf = deque(maxlen=SMOOTHING_WINDOW)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None
        if res.pose_landmarks:
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        x = preprocess(frame)
        conf, label = predict(x)
        if conf < CONFIDENCE_THRESHOLD: label = "Uncertain/Waiting"
        buf.append(label)
        smooth = max(set(buf), key=buf.count)
        fb = feedback(smooth, lm, w, h)
        cv2.putText(frame, f"Pose: {smooth}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y = 120
        for line in fb:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y += 30
        cv2.imshow("Yoga Pose Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

