from fastapi import FastAPI
from ultralytics import YOLO
from core.networks import FFNet
from models.request import Request
from models.response import Response
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

app = FastAPI()
model = YOLO("yolov8n.pt")
classes = [1, 2, 3, 5, 7]
ffnet = FFNet()
ffnet.load_state_dict(torch.load("SHA_model.pth", map_location=device))
ffnet.eval()


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
