from typing import Any, Callable, Optional

from rx.core import Observable
from rx.core.typing import Disposable, Scheduler, Observer

from src.utils.rx.typing import Emitter


def _act(emitter: Optional[Emitter] = None) \
        -> Callable[[Observable], Observable]:

    _emitter = emitter

    def act(source: Observable) -> Observable:
        def subscribe(obv: Observer, scheduler: Scheduler = None) -> Disposable:
            def on_next(_value: Any) -> None:
                try:
                    value = _emitter()
                except Exception as err:  # pylint: disable=broad-except
                    obv.on_error(err)
                else:
                    obv.on_next(value)

            return source.subscribe_(
                on_next, obv.on_error, obv.on_completed, scheduler)
        return Observable(subscribe)
    return act
