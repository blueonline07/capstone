## Vehicle Counting App (YOLOv8 + Supervision)

Python CLI to count vehicles crossing a user-defined line in a video. Uses Ultralytics YOLOv8 for detection, Supervision for tracking/line counting, and OpenCV for video I/O.

### Setup

1) Create and activate a virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

### Usage

Count on a video with default COCO weights and a horizontal mid-screen line:
```bash
python -m vehicle_counter.count --video path/to/video.mp4 --output out.mp4
```

Custom line and custom weights:
```bash
python -m vehicle_counter.count --video city.mp4 --output city_out.mp4 \
  --line 100 400 1100 400 --weights models/yolov8n-moto.pt \
  --classes car truck bus motorcycle
```

Optional flags:
- `--conf 0.25` confidence threshold (default 0.25)
- `--imgsz 640` inference image size (default 640)

The script writes an annotated video and prints counts at the end like:
```
{"in": 23, "out": 17}
```

### Notes

- Default classes counted: `car truck bus motorcycle`. You can customize with `--classes`.
- If CUDA is available, Ultralytics will leverage GPU automatically.

---

## Fine-tuning the Model for Motorcycles

Goal: Improve recall/precision for motorcycles in your domain (angles, lighting, density).

### Prepare the dataset

1) Collect images or sample frames from target videos emphasizing motorcycles.
2) Label in YOLO format (Label Studio, Roboflow, CVAT). Recommended multi-class:
```yaml
path: datasets/motorcycle
train: images/train
val: images/val
names: [car, truck, bus, motorcycle]
```
3) Split: 70/20/10 train/val/test.

### Train (Ultralytics)

Start from COCO-pretrained weights for transfer learning:
```bash
yolo detect train model=yolov8n.pt data=datasets/motorcycle.yaml \
  epochs=60 imgsz=640 batch=16 device=0 lr0=0.01 \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 degrees=5 translate=0.1 scale=0.5 shear=0.0 \
  mosaic=1.0 mixup=0.1 copy_paste=0.1 fliplr=0.5
```

Monitor mAP and per-class precision/recall, especially `motorcycle`.

### Active learning (hard negatives)

- Run the current model on new traffic videos.
- Collect false negatives/positives for motorcycles, relabel, and add to training set.
- Retrain for a few epochs to converge.

### Use trained weights in this app

1) After training, pick best weights: `runs/detect/train/weights/best.pt`.
2) Copy to `models/yolov8n-moto.pt`.
3) Run the counter with:
```bash
python -m vehicle_counter.count --video your.mp4 --output out.mp4 --weights models/yolov8n-moto.pt
```

### Tips

- Consider `imgsz=960` and multi-scale to help small motorcycles.
- If bicycles/scooters get overcounted, refine labels or use stricter class-wise thresholds.
- For higher accuracy on small objects, try `yolov8s.pt` as a base model.


