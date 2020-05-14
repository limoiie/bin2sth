from typing import Callable, Optional

from rx.core import Observable
from rx.core.typing import Predicate, Mapper

from src.utils.rx.typing import Tapper, Emitter


def tap(tapper: Optional[Tapper] = None) -> \
        Callable[[Observable], Observable]:
    from src.utils.rx.core.operators.tap import _tap
    return _tap(tapper)


def act(emitter: Optional[Emitter] = None) -> \
        Callable[[Observable], Observable]:
    from src.utils.rx.core.operators.act import _act
    return _act(emitter)


def end_act(emitter: Optional[Emitter] = None,
            predicate: Optional[Predicate] = None) -> \
        Callable[[Observable], Observable]:
    from src.utils.rx.core.operators.end_act import _end_act
    return _end_act(emitter, predicate)


def end_map(mapper: Optional[Mapper] = None,
            predicate: Optional[Predicate] = None) -> \
        Callable[[Observable], Observable]:
    from src.utils.rx.core.operators.end_act import _end_act
    return _end_act(mapper, predicate)
