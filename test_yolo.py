
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "assets" / "sample_images" / "forest.jpeg"
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo_exploration"

# Load a larger pretrained YOLO model
model = YOLO(YOLO_MODEL_PATH)

results = model.predict(
    source=IMAGE_PATH,
    classes=[0],
    conf=0.1,
    imgsz=960,
    save=True,
    project=OUTPUT_DIR,
    name="forest_test",
    exist_ok=True
)

for result in results:
    print("People detected:", len(result.boxes))

    for box in result.boxes:
        print("Confidence:", round(float(box.conf[0]), 3))
