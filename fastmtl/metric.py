# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_metric.ipynb (unless otherwise specified).

__all__ = ['RoutedAccumMetric', 'route_to_metric', 'mtl_metrics']

# Cell

from fastcore.basics import GetAttr, store_attr
from types import FunctionType
from fastai.metrics import Metric, AvgMetric, AccumMetric


class _LearnerProxy(GetAttr):
    _default = 'learn'
    def __init__(self, learn, idx):
        store_attr()
        self.pred = self.learn.pred[idx]
        self.y = self.learn.y[idx]


class RoutedAccumMetric(AccumMetric, GetAttr):
    "AccumMetric with predictions and targets for a specific model head."
    _default = 'metric'
    def __init__(self, idx, metric):
        self.idx = idx
        self.metric = metric
        self._name = metric.name

    def reset(self):
        "Clear all targs and preds"
        return self.metric.reset()

    def accumulate(self, learn):
        "Store targs and preds from `learn`, using activation function and argmax as appropriate"
        return self.metric.accumulate(_LearnerProxy(learn, self.idx))

    def __call__(self, preds, targs):
        "Calculate metric on one batch of data"
        return self.metric(preds[self.idx], targs[self.idx])

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

# Cell

def route_to_metric(idx, metric):
    """Routes model output at idx to metric"""
    if isinstance(metric, type):
        metric = metric()
    if isinstance(metric, FunctionType):
        func = lambda preds, *targs, **kwargs: metric(preds[idx], targs[idx], **kwargs)
        func.__name__ = metric.__name__
        return AvgMetric(func)
    if isinstance(metric, Metric):
        return RoutedAccumMetric(idx, metric)
    raise ValueError("Unsupported metric type; must be either function or Metric")


# Cell

def mtl_metrics(*metrics_list):
    """Convenience function to route each prediction to list of metrics by their order."""
    return [route_to_metric(i, m) for i, metrics in enumerate(metrics_list) for m in metrics]