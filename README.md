# Traffic Volume API

This project is a FastAPI-based application for counting vehicles in a series of images using computer vision models.

## Description

The application provides a `/process` endpoint that accepts a list of image frames and returns a vehicle count. It uses a two-stage process for efficiency:

1.  A lightweight custom model (`FFNet`) is first used to get an initial count.
2.  If the count from `FFNet` is below a certain threshold, a more powerful YOLOv8 model is used for a more accurate count of specific vehicle classes.

This approach allows for fast processing of low-density traffic while retaining the accuracy of a larger model for more complex scenes.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Download the pre-trained model weights:
    -   `yolov8n.pt`
    -   `SHA_model.pth`
    Place them in the root of the project directory.

## Usage

1.  Start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```

2.  Send a POST request to the `/process` endpoint.

    **URL:** `http://127.0.0.1:8000/process`

    **Body:**

    The request body should be a JSON object with the following structure (as defined in `models/request.py`):

    ```json
    {
      "camera_id": "string",
      "slot": "string",
      "generated_at": "2025-12-06T12:00:00Z",
      "duration_sec": 0,
      "speed": 0,
      "weather": "string",
      "frames": [
        {
          "image_ref": "path/to/your/image1.jpg",
          "timestamp": "2025-12-06T12:00:00Z"
        },
        {
          "image_ref": "path/to/your/image2.jpg",
          "timestamp": "2025-12-06T12:00:01Z"
        }
      ]
    }
    ```

    **Note:** The `image_ref` should be a path to an image file that the application can access. The current implementation in `main.py` uses local file paths.

    **Example using `curl`:**

    ```bash
    curl -X POST "http://127.0.0.1:8000/process" -H "Content-Type: application/json" -d '{
      "camera_id": "cam-01",
      "slot": "A",
      "generated_at": "2025-12-06T12:00:00Z",
      "duration_sec": 5,
      "frames": [
        {"image_ref": "path/to/image1.jpg", "timestamp": "2025-12-06T12:00:00Z"},
        {"image_ref": "path/to/image2.jpg", "timestamp": "2025-12-06T12:00:01Z"}
      ]
    }'
    ```

    The server will respond with a JSON object containing the vehicle count.

## Configuration

The application can be configured via environment variables. The following variables are available (defined in `config/settings.py`):

-   `APP_NAME`: The name of the application. Defaults to `"Traffic Volume API"`.
-   `THRESHOLD`: The threshold for switching between the `FFNet` and YOLOv8 models. If the `FFNet` count is below this value, YOLOv8 is used. Defaults to `128`.

To set an environment variable, you can use:

```bash
export THRESHOLD=100
uvicorn main:app --reload
```
