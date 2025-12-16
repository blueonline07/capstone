# Traffic Volume API

FastAPI application for counting vehicles in images using YOLO.

## Features

- **API Endpoint**: Process frames and count vehicles via `/process`
- **Batch Processing**: Process local images by date and store results in NPY format

## Installation

```bash
# Clone repository
git clone https://github.com/blueonline07/capstone
cd capstone

# Install dependencies
pip install -r requirements.txt
```

## Usage

### API Server

Start the server:

```bash
uvicorn main:app --reload
```

Send POST request to `/process`:

```bash
curl -X POST "http://127.0.0.1:8000/process" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "cam_001",
    "slot": "00:00:00",
    "generated_at": "2025-11-15T00:00:00",
    "duration_sec": 300,
    "frames": [
      {"time": "2025-11-15T00:00:00", "image_ref": "path/to/image.jpg"}
    ],
    "speed": null,
    "weather": null
  }'
```

### Batch Processing

Process local images by date from folder structure:

```bash
python process_batch.py --date 2025-11-15
```

**Input structure:** `batch/<node>/batch_<slot>.json`

**Output:** `output/<date>/<slot>.npy` containing `[response_of_node1, response_of_node2, ...]`

Each slot file across all nodes is processed and combined into one output file per slot.

## Configuration

Environment variables (see `config/settings.py`):

- `APP_NAME`: Application name (default: "Traffic Volume API")

## Models

The application uses YOLO11n (`yolo11n.pt`) to detect vehicles with the following classes:
- Class 1: Bicycle
- Class 2: Car
- Class 3: Motorcycle
- Class 5: Bus
- Class 7: Truck

## Project Structure

```
├── main.py                    # FastAPI application
├── process_batch.py           # Batch processing script
├── models/                    # Pydantic models
│   ├── request.py
│   ├── response.py
│   └── common.py
├── config/                    # Configuration
│   └── settings.py
├── batch/                     # Input folder
│   └── <node>/
│       └── batch_<slot>.json
└── output/                    # Batch processing results
    └── <date>/
        └── <slot>.npy
```
