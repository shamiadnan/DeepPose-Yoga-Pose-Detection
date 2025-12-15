import torch
from pathlib import Path

CKPT_DIR = Path("checkpoints")
LOG_DIR  = Path("logs")

CLASSES = ["downdog", "goddess", "plank", "tree", "warrior"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_WINDOW = 5

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
