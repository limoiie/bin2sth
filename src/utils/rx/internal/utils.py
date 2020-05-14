import queue

from rx import Observable, operators as ops
from rx.internal import noop
from rx.scheduler import NewThreadScheduler

from src.utils.rx import operators as ops_


class ObsIterable:
    _scheduler = NewThreadScheduler()

    class Iter:
        def __init__(self, obs: Observable):
            self.pipe = queue.Queue()
            obs.pipe(
                ops_.tap(self.pipe.put),
                ops.last(), ops.map(noop),
                ops_.tap(self.pipe.put),
                ops.subscribe_on(ObsIterable._scheduler)
            ).run()

        def __next__(self):
            x = self.pipe.get()
            if x is None:
                raise StopIteration()
            return x

    def __init__(self, obs: Observable):
        self.obs = obs

    def __iter__(self):
        return ObsIterable.Iter(self.obs)