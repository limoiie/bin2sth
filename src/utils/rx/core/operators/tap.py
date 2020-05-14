from typing import Any, Callable, Optional

from rx.core import Observable
from rx.core.typing import Disposable, Scheduler, Observer
from rx.internal import noop

from src.utils.rx.typing import Tapper


def _tap(tapper: Optional[Tapper] = None) -> Callable[[Observable], Observable]:

    _tapper = tapper or noop

    def tap(source: Observable) -> Observable:
        def subscribe(obv: Observer, scheduler: Scheduler = None) -> Disposable:
            def on_next(value: Any) -> None:
                try:
                    _tapper(value)
                except Exception as err:  # pylint: disable=broad-except
                    obv.on_error(err)
                else:
                    obv.on_next(value)

            return source.subscribe_(
                on_next, obv.on_error, obv.on_completed, scheduler)
        return Observable(subscribe)
    return tap
