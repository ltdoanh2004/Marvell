# src/models.py
from dataclasses import dataclass

@dataclass
class TravelQuery:
    destination: str

@dataclass
class TravelResponse:
    content: str
    source: str  # "destination" or "flight"