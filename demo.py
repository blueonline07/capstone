import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import torch
from PIL import Image
from networks.FFNet import FFNet
from torchvision import transforms
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FFNet/YOLO inference on images.")
    parser.add_argument(
        "--node",
        type=str,
        required=True,
        help="Node/camera ID whose frames will be processed",
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to process (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default="batch",
        help="Root directory containing node batch folders (default: batch)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Threshold for automatic model selection (default: 15.0)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO model weights (default: yolov8n.pt)",
    )
    args = parser.parse_args()

    output_dir = Path("output") / args.date
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"{args.node}.npy"

    checkpoint = torch.load("SHA_model.pth", map_location="cpu")

    ffnet_model = FFNet()
    ffnet_model.load_state_dict(checkpoint)
    ffnet_model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ffnet_model.to(device)

    # Initialize YOLO model
    yolo_model = YOLO(args.yolo_weights)

    # Vehicle class IDs in COCO dataset: car=2, motorcycle=3, bus=5, truck=7
    vehicle_classes = [2, 3, 5, 7]

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    slot_summaries: dict[str, dict[str, object]] = {}
    if progress_path.exists():
        print(f"Loading existing progress from {progress_path}")
        existing_data = np.load(progress_path, allow_pickle=True)
        loaded_count = 0
        for entry in existing_data:
            slot_value = entry.get("slot") if isinstance(entry, dict) else None
            if slot_value:
                slot_summaries[str(slot_value)] = dict(entry)
                loaded_count += 1
        print(f"Loaded {loaded_count} existing slots")

    def run_yolo_inference(image_path: Path) -> tuple[float, int]:
        """Run YOLO inference and count vehicles in the image.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple of (inference_time, vehicle_count)
        """
        image = Image.open(image_path).convert("RGB")
        base = time.perf_counter()
        results = yolo_model(image)
        elapsed = time.perf_counter() - base

        # Count vehicles from all detections
        vehicle_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Filter for vehicle classes
                for cls in boxes.cls:
                    if int(cls) in vehicle_classes:
                        vehicle_count += 1

        return elapsed, vehicle_count

    def run_inference_on_image(image_path: Path) -> Optional[dict[str, object]]:
        """Run inference on an image using automatic model selection.

        FFNet is run first. If count > threshold, use FFNet result.
        Otherwise, run YOLO and use YOLO result.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return None

        # Run FFNet first to get initial count
        input_tensor = transform(image).unsqueeze(0).to(device)
        base = time.perf_counter()
        with torch.no_grad():
            output, _ = ffnet_model(input_tensor)
        ffnet_elapsed = time.perf_counter() - base
        ffnet_count = torch.sum(output).item()

        # Automatic model selection based on threshold
        if ffnet_count > args.threshold:
            # Dense traffic: use FFNet
            model_used = "FFNet"
            final_count = ffnet_count
            inference_time = ffnet_elapsed
        else:
            # Sparse traffic: use YOLO
            yolo_elapsed, yolo_count = run_yolo_inference(image_path)
            model_used = "YOLO"
            final_count = yolo_count
            inference_time = yolo_elapsed

        return {
            "image_path": str(image_path),
            "model": model_used,
            "count": float(final_count),
            "inference_time": float(inference_time),
        }

    def run_inference_for_node_day(node: str, date_str: str, batch_root: Path) -> None:
        """Run inference across all frames for a node on a specific day."""

        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"Invalid --date value '{date_str}'. Expected format YYYY-MM-DD."
            ) from exc

        node_dir = batch_root / node
        if not node_dir.exists() or not node_dir.is_dir():
            raise ValueError(f"Node directory does not exist: {node_dir}")

        json_files = sorted(node_dir.glob("batch_*.json"))
        if not json_files:
            print(f"No batch files found for node {node}")
            return

        print(f"Processing {len(json_files)} batch files for node {node}")
        processed_slots = 0
        skipped_slots = 0

        for json_path in json_files:
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except json.JSONDecodeError:
                continue

            slot = str(payload.get("slot") or json_path.stem.replace("batch_", ""))
            if not slot.startswith(date_str):
                continue
            if slot in slot_summaries:
                skipped_slots += 1
                continue

            print(f"Processing slot {slot}...", end=" ", flush=True)
            slot_summary = slot_summaries.get(slot)
            if slot_summary is None:
                slot_summary = {
                    key: value for key, value in payload.items() if key != "frames"
                }
                slot_summary["vehicles_count"] = 0.0
                slot_summaries[slot] = slot_summary

            frames = payload.get("frames", [])
            frame_count = 0
            for frame in frames:
                image_ref = frame.get("image_ref")
                if not image_ref:
                    continue
                result = run_inference_on_image(Path(image_ref))
                if result is None:
                    continue
                slot_summary["vehicles_count"] += float(result["count"])
                frame_count += 1

            processed_slots += 1
            print(
                f"done ({frame_count} frames, {slot_summary['vehicles_count']:.1f} vehicles)"
            )

        if skipped_slots > 0:
            print(f"Skipped {skipped_slots} already processed slots")
        print(
            f"Total: {processed_slots} new slots processed, {len(slot_summaries)} total slots"
        )

    print(f"\n{'='*60}")
    print(f"Processing node: {args.node}, date: {args.date}")
    print(f"{'='*60}\n")

    batch_root = Path(args.batch_root)
    run_inference_for_node_day(args.node, args.date, batch_root)

    slot_array = [slot_summaries[key] for key in sorted(slot_summaries.keys())]

    print(f"Saving {len(slot_array)} slots to {progress_path}")
    np.save(
        progress_path,
        np.array(slot_array, dtype=object),
        allow_pickle=True,
    )
    print("Saved successfully")
