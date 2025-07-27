# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

from ultralytics import YOLO

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODEL_PATH = "/home/nvidia/Downloads/yolov8s.pt"
DATA_PATH = "/home/nvidia/new_rec2/followmanta_sesssion/mantaDataset/manta20240829.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH = 1
DEVICE = "cuda"

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
model = YOLO(MODEL_PATH)  # Load a pretrained model
model.to(DEVICE)  # Move model to GPU

# Train the model
results = model.train(data=DATA_PATH, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH)