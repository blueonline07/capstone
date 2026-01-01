from pydantic import BaseModel
from typing import List, Optional


class SpeedEntry(BaseModel):
    time: str
    speed: float


class Speed(BaseModel):
    count: int
    avg_kmh: float
    min_kmh: float
    max_kmh: float
    series: List[SpeedEntry]


class Coord(BaseModel):
    lon: float
    lat: float


class WeatherDescription(BaseModel):
    id: int
    main: str
    description: str
    icon: str


class MainWeather(BaseModel):
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: int
    humidity: int
    sea_level: Optional[int] = None
    grnd_level: Optional[int] = None


class Wind(BaseModel):
    speed: float
    deg: int


class Clouds(BaseModel):
    all: int


class Sys(BaseModel):
    type: Optional[int] = None
    id: Optional[int] = None
    country: str
    sunrise: int
    sunset: int


class Weather(BaseModel):
    coord: Coord
    weather: List[WeatherDescription]
    base: str
    main: MainWeather
    visibility: int
    wind: Wind
    clouds: Clouds
    dt: int
    sys: Sys
    timezone: int
    id: int
    name: str
    cod: int
