from pydantic import BaseModel
from models.common import Speed, Weather
from typing import Optional


class Response(BaseModel):
    camera_id: str
    slot: str
    generated_at: str
    duration_sec: int
    vehicles_count: float
    speed: Optional[Speed]
    weather: Optional[Weather]
