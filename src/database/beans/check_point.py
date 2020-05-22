from dataclasses import dataclass
from typing import Any

from src.utils.auto_json import auto_json


@auto_json
@dataclass
class CheckPointEntry:
    training_id: str = ''
    name: str = ''
    data: Any = None
