"""
Process batch data from folder structure: batch/<node>/<slot>.json
Output: output/<date>/<slot>.npy containing [response_of_node1, response_of_node2, ...]
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from models.request import Request
from models.response import Response
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_slot(args):
    """Process a single slot across all nodes."""
    slot_time, node_files, model_path, output_dir, date = args

    # Load model (each process gets its own model instance)
    model = YOLO(model_path, verbose=False)
    classes = [1, 2, 3, 5, 7]  # Vehicle classes

    responses = []

    # Process each node for this slot
    for node_id, slot_file in node_files:
        try:
            # Load request from JSON file
            with open(slot_file, "r") as f:
                req = Request(**json.load(f))

            # Run YOLO on frames
            image_refs = [frame.image_ref for frame in req.frames]
            results = model(image_refs, classes=classes, verbose=False)
            vehicles_count = sum([len(r.boxes) for r in results])

            # Create response
            response = Response(
                camera_id=req.camera_id,
                slot=req.slot,
                generated_at=req.generated_at,
                duration_sec=req.duration_sec,
                vehicles_count=vehicles_count,
                speed=req.speed,
                weather=req.weather,
            )

            responses.append(response.model_dump())

        except Exception as e:
            print(f"\nError processing {node_id}/{slot_file.name}: {e}")
            continue

    # Save all node responses for this slot
    if responses:
        output_date_dir = Path(output_dir) / date
        output_file = output_date_dir / f"{slot_time}.npy"
        np.save(output_file, responses, allow_pickle=True)

    return slot_time


def process_batch(
    date: str,
    batch_dir: str = "batch",
    output_dir: str = "output",
    model_path: str = "yolo11n.pt",
    workers: int = None,
):
    """Process all slots for a given date across all nodes."""

    # Get all node directories
    batch_path = Path(batch_dir)
    node_dirs = [d for d in batch_path.iterdir() if d.is_dir()]

    if not node_dirs:
        print(f"Error: No node directories found in {batch_dir}")
        return

    print(f"Found {len(node_dirs)} nodes")

    # Collect all slots grouped by slot timestamp
    slots_by_time = defaultdict(list)  # {slot_time: [(node, file_path), ...]}

    for node_dir in node_dirs:
        node_id = node_dir.name

        # Find slot files for this date
        for slot_file in node_dir.glob(f"batch_{date}*.json"):
            # Extract slot time from filename: batch_2025-11-12T11:10:00.json -> 2025-11-12T11:10:00
            slot_time = slot_file.stem.replace("batch_", "")
            slots_by_time[slot_time].append((node_id, slot_file))

    if not slots_by_time:
        print(f"No slots found for date {date}")
        return

    print(f"Found {len(slots_by_time)} unique slots")

    # Create output directory
    output_date_dir = Path(output_dir) / date
    output_date_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    tasks = [
        (slot_time, node_files, model_path, output_dir, date)
        for slot_time, node_files in sorted(slots_by_time.items())
    ]

    # Determine number of workers
    if workers is None:
        workers = min(cpu_count(), len(tasks))

    print(f"Processing with {workers} workers")

    # Process slots in parallel
    with Pool(processes=workers) as pool:
        list(
            tqdm(
                pool.imap(process_slot, tasks),
                total=len(tasks),
                desc=f"Processing {date}",
            )
        )

    print(f"\nResults saved to: {output_dir}/{date}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process batch data by date")
    parser.add_argument("--date", required=True, help="Date to process (YYYY-MM-DD)")
    parser.add_argument("--batch-dir", default="batch", help="Batch input directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)",
    )

    args = parser.parse_args()
    process_batch(args.date, args.batch_dir, args.output_dir, args.model, args.workers)
