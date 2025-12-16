from fastapi import FastAPI
from ultralytics import YOLO
from models.request import Request
from models.response import Response
from config.settings import settings

app = FastAPI()
model = YOLO("yolo11n.pt")
classes = [1, 2, 3, 5, 7]


@app.post("/process")
async def process(req: Request) -> Response:
    # TODO: replace image_ref with remote url
    results = model([e.image_ref for e in req.frames], classes=classes)
    cnt = sum([len(r.boxes) for r in results])

    return Response(
        camera_id=req.camera_id,
        slot=req.slot,
        generated_at=req.generated_at,
        duration_sec=req.duration_sec,
        vehicles_count=cnt,
        speed=req.speed,
        weather=req.weather,
    )
