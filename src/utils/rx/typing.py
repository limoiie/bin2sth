from typing import Callable
from rx.core.typing import T1

Tapper = Callable[[T1], None]
Emitter = Callable[[], T1]
