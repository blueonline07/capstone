import cv2
import numpy as np
import random
import torch
from ultralytics import YOLO

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 640
HUMAN_SIZE = 40  # size of each human (px)
CONF_THRES = 0.001  # expose confidence collapse
IOU_THRES = 0.7
RECALL_COLLAPSE = 0.7  # collapse condition

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Load assets
# -----------------------------
bg = cv2.imread("background.png")
bg = cv2.resize(bg, (IMG_SIZE, IMG_SIZE))

human = cv2.imread("human.png", cv2.IMREAD_UNCHANGED)
human = cv2.resize(human, (HUMAN_SIZE, HUMAN_SIZE))


# -----------------------------
# Alpha blending
# -----------------------------
def alpha_blend(img, obj, x, y):
    h, w = obj.shape[:2]
    alpha = obj[:, :, 3] / 255.0

    for c in range(3):
        img[y : y + h, x : x + w, c] = (
            alpha * obj[:, :, c] + (1 - alpha) * img[y : y + h, x : x + w, c]
        )
    return img


# -----------------------------
# Generate crowded image
# -----------------------------
def generate_image(k):
    img = bg.copy()

    for _ in range(k):
        x = random.randint(0, IMG_SIZE - HUMAN_SIZE)
        y = random.randint(0, IMG_SIZE - HUMAN_SIZE)
        img = alpha_blend(img, human, x, y)

    return img


# -----------------------------
# Density sweep
# -----------------------------
collapse_k = None

print("Density stress test (human crowding)")
print("===================================")
print(" k | detections | recall")

for k in range(10, 1000, 10):
    img = generate_image(k)

    results = model.predict(source=img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)

    boxes = results[0].boxes
    detections = len(boxes) if boxes is not None else 0
    recall = detections / k

    print(f"{k:3d} | {detections:10d} | {recall:.3f}")

    if recall < RECALL_COLLAPSE and collapse_k is None:
        collapse_k = k
        print(f"\n🚨 COLLAPSE DETECTED at k = {k}\n")
        break

if collapse_k is None:
    print("\n✅ No collapse detected in tested range")
