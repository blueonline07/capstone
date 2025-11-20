# Vehicle Counting System (FFNet + YOLO)

A Python application for vehicle counting using a hybrid approach that combines FFNet (for dense traffic) and YOLO (for sparse traffic) models. The system processes batch JSON files containing image references and automatically selects the appropriate model based on traffic density.

## Setup

### Prerequisites

- Python 3.8 or higher
- Model files:
  - `SHA_model.pth` - FFNet model weights (should be in the project root)
  - `yolov8n.pt` - YOLO model weights (will be downloaded automatically if not present)

### Installation

1. **Create and activate a virtual environment** (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install torch torchvision pillow
```

Note: If you have a CUDA-capable GPU, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Running the Application

### Basic Usage

Process a single node for a specific date:
```bash
python demo.py --node <node_id> --date YYYY-MM-DD
```

Example:
```bash
python demo.py --node 58af8a0bbd82540010390c25 --date 2025-11-13
```

### Command Line Arguments

- `--node` (required): Node/camera ID whose frames will be processed
- `--date` (required): Date to process in YYYY-MM-DD format
- `--batch-root` (optional): Root directory containing node batch folders (default: `batch`)
- `--threshold` (optional): Threshold for automatic model selection (default: 15.0)
  - If FFNet count > threshold, use FFNet result
  - Otherwise, use YOLO result
- `--yolo-weights` (optional): Path to YOLO model weights (default: `yolov8n.pt`)

### Example with Custom Options

```bash
python demo.py \
  --node 58af8a0bbd82540010390c25 \
  --date 2025-11-13 \
  --batch-root batch \
  --threshold 20.0 \
  --yolo-weights yolov8n.pt
```

### Processing All Nodes

Use the provided shell script to process all nodes in parallel:
```bash
export DATE=2025-11-13
export CONCURRENCY=4  # Number of parallel processes (optional, default: 4)
./run_all_nodes.sh
```

Or run directly:
```bash
DATE=2025-11-13 CONCURRENCY=4 ./run_all_nodes.sh
```

## Project Structure

```
.
├── demo.py              # Main inference script
├── SHA_model.pth        # FFNet model weights
├── yolov8n.pt          # YOLO model weights
├── requirements.txt     # Python dependencies
├── run_all_nodes.sh    # Script to process all nodes
├── batch/              # Directory containing node batch folders
│   └── <node_id>/      # Node-specific batch JSON files
│       └── batch_*.json
├── networks/           # Neural network definitions
│   ├── FFNet.py       # FFNet model architecture
│   └── ODConv2d.py    # ODConv2d layer implementation
└── output/            # Output directory for results
    └── YYYY-MM-DD/    # Date-specific output folders
        └── <node_id>.npy
```

## How It Works

1. **Model Selection**: The system runs FFNet first to get an initial vehicle count. If the count exceeds the threshold (default: 15.0), it uses the FFNet result. Otherwise, it runs YOLO for more accurate sparse traffic detection.

2. **Processing**: The script processes batch JSON files for the specified node and date. Each JSON file contains:
   - Metadata about the time slot
   - References to image files to process

3. **Output**: Results are saved as NumPy arrays (`.npy` files) in the `output/` directory, organized by date and node ID. The script supports resuming - it will skip already processed slots.

4. **Device Selection**: The system automatically uses the best available device:
   - MPS (Apple Silicon GPU) if available
   - CUDA (NVIDIA GPU) if available
   - CPU otherwise

## Notes

- The system counts vehicles from COCO classes: car (2), motorcycle (3), bus (5), and truck (7)
- Progress is saved incrementally, so you can safely interrupt and resume processing
- Output files are saved in NumPy format with pickle support for complex data structures
- The batch directory should contain subdirectories named with node IDs, each containing `batch_*.json` files
