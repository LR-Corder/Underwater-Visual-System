from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("/home/nvidia/Downloads/yolov8s.pt")  # load a pretrained model (recommended for training)
model.to("cuda")
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/home/nvidia/new_rec2/followmanta_sesssion/mantaDataset/manta20240829.yaml", epochs=100, imgsz=640,batch=1)
