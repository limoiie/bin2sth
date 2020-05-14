from typing import Callable, Optional

import rx.operators as ops
from rx.core import Observable
from rx.core.typing import Predicate, Mapper
from rx.internal.basic import identity


def _end_act(mapper: Optional[Mapper] = None,
             predicate: Optional[Predicate] = None) \
        -> Callable[[Observable], Observable]:

    _mapper = mapper or identity

    def end_map(source: Observable) -> Observable:
        return source.pipe(
            ops.last(predicate),
            ops.map(_mapper)
        )

    return end_map
