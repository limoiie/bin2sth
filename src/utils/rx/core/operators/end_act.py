from typing import Callable, Optional

import rx.operators as ops
from rx.core import Observable
from rx.core.typing import Predicate

import src.utils.rx.operators as ops_
from src.utils.rx.typing import Emitter


def _end_act(emitter: Optional[Emitter] = None,
             predicate: Optional[Predicate] = None) \
        -> Callable[[Observable], Observable]:

    def end_act(source: Observable) -> Observable:
        return source.pipe(
            ops.last(predicate),
            ops_.act(emitter)
        )

    return end_act
