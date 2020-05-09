from ignite.metrics import Metric, EpochMetric


class SiameseMetric(Metric):
    """
    Wrap the metric for the model which is under the siamse architecture,
    where the similarity has not been evaluated. This will compute the
    sim between the two outputs before performing the target metric
    :param sim_fn: to compute the similarity by given two outputs
    :param metric: target metric which is supposed to work on the output
    that in the form of `(y_pred, y)'
    """
    def __init__(self, sim_fn, metric: Metric):
        self.metric = metric
        super().__init__()
        assert (issubclass(type(metric), Metric) or
                issubclass(type(metric), EpochMetric)), \
            ':param metric should be instance of :class Metric!'
        self.sim_fn = sim_fn

    def reset(self):
        self.metric.reset()

    def update(self, output):
        (o1, o2), y = output
        output = self.sim_fn(o1, o2), y
        return self.metric.update(output)

    def compute(self):
        return self.metric.compute()
