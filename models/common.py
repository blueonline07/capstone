from pydantic import BaseModel
from typing import List

class SpeedEntry(BaseModel):
    time: str
    speed: float

class Speed(BaseModel):
    count: int
    avg_kmh: float
    min_kmh: float
    max_kmh: float
    series: List[SpeedEntry]