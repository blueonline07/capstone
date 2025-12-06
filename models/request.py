from typing import List, Optional
from pydantic import BaseModel
from models.common import Speed


class Frame(BaseModel):
    time: str
    image_ref: str


class Request(BaseModel):
    camera_id: str
    slot: str
    generated_at: str
    duration_sec: int
    frames: List[Frame]
    speed: Optional[Speed]
    weather: Optional[str]
