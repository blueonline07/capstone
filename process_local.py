import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from networks.wrapper import FFNetWrapper
from models.request import Request
from models.response import Response as ResponseModel
from config.settings import settings


def process_local(batch_dir: str, output_dir: str, date: str = None):
    """
    Reads JSON files from the batch directory, runs the model directly,
    and saves the responses grouped by timestamp to a NumPy file.
    """
    json_files = sorted(list(Path(batch_dir).rglob("*.json")))
    if not json_files:
        print(f"No JSON files found in {batch_dir}")
        return

    if date:
        json_files = [f for f in json_files if f.stem.startswith(f"batch_{date}")]

    # Group files by timestamp
    grouped_files = defaultdict(list)
    for json_file in json_files:
        timestamp = json_file.stem.replace("batch_", "")
        grouped_files[timestamp].append(json_file)

    Path(output_dir).mkdir(exist_ok=True)

    # Load models
    model = YOLO("yolov8n.pt")
    ffnet = FFNetWrapper("SHA_model.pth")
    classes = [1, 2, 3, 5, 7]

    for timestamp, files in tqdm(grouped_files.items(), desc="Processing timestamps"):
        responses = []
        for json_file in files:
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                    req = Request(**data)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Skipping invalid JSON file: {json_file}, error: {e}")
                    continue

            # This is the logic from the process function in main.py
            cnt = ffnet.predict([e.image_ref for e in req.frames])
            if cnt < settings.threshold:
                results = model(
                    [e.image_ref for e in req.frames], classes=classes, verbose=False
                )
                cnt = sum([len(r.boxes) for r in results])

            response = ResponseModel(
                camera_id=req.camera_id,
                slot=req.slot,
                generated_at=req.generated_at,
                duration_sec=req.duration_sec,
                vehicles_count=cnt,
                speed=req.speed,
                weather=req.weather,
            )
            responses.append(response.model_dump())

        if responses:
            output_file = Path(output_dir) / f"{timestamp}.npy"
            np.save(output_file, np.array(responses, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process batch data locally.")
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="batch",
        help="The directory containing the JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_local",
        help="The directory to save the responses to.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="The date to process files for, in YYYY-MM-DD format.",
    )
    args = parser.parse_args()

    process_local(args.batch_dir, args.output_dir, args.date)
