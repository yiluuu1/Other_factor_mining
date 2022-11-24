#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/9 11:04
# @Author : Yilu Jiang
# @File : my_functions.py
"""The functions used to create programs.
"""

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects

__all__ = ['make_function', '_Function']


class _Function(object):
    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity, extra_param=None):
        self.function = function
        self.name = name
        self.arity = arity
        self.extra_param = extra_param

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, extra_param):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.
    extra_param: tuple
        The extra param for some function
    """
    args = [np.ones(10) for _ in range(arity)]
    function(*args)
    return _Function(function=wrap_non_picklable_objects(function), name=name, arity=arity, extra_param=extra_param)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    res = x1 / x2
    res[np.isinf(res)] = (res[~np.isinf(res)]).mean()
    return res


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore'):
        res = np.log(abs(x1))
        res[np.isinf(res)] = (res[~np.isinf(res)]).mean()
        return res


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    return _protected_division(1, x1)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    return 1 / (1 + np.exp(-x1))


def _cross_sectional_rank(x1):
    return x1.groupby('date', group_keys=False).rank(pct=True)


def _cross_sectional_scale(x1):
    """Scaling time serie."""

    def scale(ft):
        if abs(ft).sum() != 0:
            return ft.div(np.abs(ft).sum())
        else:
            return ft

    return x1.groupby('date', group_keys=False).apply(scale)


def _ts_delay(x1, window):
    return x1.groupby('code', group_keys=False).shift(window)


def _ts_delta(x1, window):
    return x1.groupby('code', group_keys=False).diff(window)


def _ts_sum(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).sum().droplevel(0)


def _ts_std(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).std().droplevel(0)


def _ts_avg(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).mean().droplevel(0)


def _ts_min(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).min().droplevel(0)


def _ts_max(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).max().droplevel(0)


def _ts_mid(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).quantile(0.5).droplevel(0)


def _ts_rank(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).rank(pct=True).droplevel(0)


def _ts_prod(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).apply(
        np.prod, raw=True, engine='numba').droplevel(0)


def _ts_argmax(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).apply(
        np.argmax, raw=True, engine='numba').droplevel(0) + 1


def _ts_argmin(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).apply(
        np.argmin, raw=True, engine='numba').droplevel(0) +1


def _ts_corr(x1, x2, window):
    def ts_corr(ft):
        res = ft.iloc[:, 0].rolling(window).corr(ft.iloc[:, 1])
        # If x1 or x2's std is 0, cov(x1, x2) will be 0, then we will get a None for 0/0
        res[np.isinf(res)] = res.mean()
        res[window - 1:][np.isnan(res)] = 0
        return res

    return pd.concat([x1, x2], axis=1).groupby('code', group_keys=False).apply(ts_corr)


def _ts_cov(x1, x2, window):
    return pd.concat([x1, x2], axis=1).groupby('code', group_keys=False).apply(
        lambda ft: ft.iloc[:, 0].rolling(window).cov(ft.iloc[:, 1]))


def _ts_decay_weight(x1, window):
    """Linear weighted moving average implementation.
       For least data's weight is d, for farest data's weight is d minus window
       Then we will scale their weight to 1"""
    weight = (np.arange(window) + 1) / (window * (window + 1) / 2)
    return x1.groupby('code', group_keys=False).rolling(window).apply(
        np.dot, args=(weight,), raw=True, engine='numba').droplevel(0)


def _ts_slope(x1, window):
    def slope(ft):
        x = np.arange(window) + 1
        return np.sum((x - np.mean(x)) * (ft - np.mean(ft))) / np.sum((x - np.mean(x)) ** 2)

    return x1.groupby('code').rolling(window).apply(slope, raw=True, engine='numba').droplevel(0)


def _ts_intercept(x1, window):
    def intercept(ft):
        x = np.arange(window) + 1
        beta = np.sum((x - np.mean(x)) * (ft - np.mean(ft))) / np.sum((x - np.mean(x)) ** 2)
        return np.mean(ft) - beta * np.mean(x)

    return x1.groupby('code').rolling(window).apply(intercept, raw=True, engine='numba').droplevel(0)


add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)
power1 = _Function(function=np.power, name='pow', arity=1, extra_param=(2, 6))
cs_rank1 = _Function(function=_cross_sectional_rank, name='cs_rank', arity=1)
scale1 = _Function(function=_cross_sectional_scale, name='scale', arity=1)
ts_delay1 = _Function(function=_ts_delay, name='ts_delay', arity=1, extra_param=(1, 60))
ts_delta1 = _Function(function=_ts_delta, name='ts_delta', arity=1, extra_param=(1, 60))
ts_corr2 = _Function(function=_ts_corr, name='ts_corr', arity=2, extra_param=(1, 60))
ts_cov2 = _Function(function=_ts_cov, name='ts_cov', arity=2, extra_param=(1, 60))
ts_rank1 = _Function(function=_ts_rank, name='ts_rank', arity=1, extra_param=(1, 60))
ts_sum1 = _Function(function=_ts_sum, name='ts_sum', arity=1, extra_param=(1, 60))
ts_prod1 = _Function(function=_ts_prod, name='ts_prod', arity=1, extra_param=(1, 60))
ts_std1 = _Function(function=_ts_std, name='ts_std', arity=1, extra_param=(1, 60))
ts_avg1 = _Function(function=_ts_avg, name='ts_avg', arity=1, extra_param=(1, 60))
ts_mid1 = _Function(function=_ts_mid, name='ts_mid', arity=1, extra_param=(1, 60))
ts_min1 = _Function(function=_ts_min, name='ts_min', arity=1, extra_param=(1, 60))
ts_max1 = _Function(function=_ts_max, name='ts_max', arity=1, extra_param=(1, 60))
ts_argmax1 = _Function(function=_ts_argmax, name='ts_argmax', arity=1, extra_param=(1, 60))
ts_argmin1 = _Function(function=_ts_argmin, name='ts_argmin', arity=1, extra_param=(1, 60))
ts_decay_weight1 = _Function(function=_ts_decay_weight, name='ts_decay_linear', arity=1, extra_param=(1, 60))
ts_slope1 = _Function(function=_ts_slope, name='ts_slope', arity=1, extra_param=(2, 60))
ts_intercept1 = _Function(function=_ts_intercept, name='ts_intercept', arity=1, extra_param=(2, 60))
_function_map = {'add': add2, 'sub': sub2, 'mul': mul2, 'div': div2, 'sqrt': sqrt1, 'log': log1, 'abs': abs1,
                 'neg': neg1, 'inv': inv1, 'max': max2, 'min': min2, 'sin': sin1, 'cos': cos1, 'tan': tan1,
                 'power': power1, 'cs_rank': cs_rank1, 'scale': scale1, 'ts_delay': ts_delay1, 'ts_delta': ts_delta1,
                 'ts_corr': ts_corr2, 'ts_cov': ts_cov2, 'ts_rank': ts_rank1, 'ts_sum': ts_sum1, 'ts_prod': ts_prod1,
                 'ts_std': ts_std1, 'ts_min': ts_min1, 'ts_max': ts_max1, 'ts_argmax': ts_argmax1,
                 'ts_argmin': ts_argmin1,
                 'ts_decay_linear': ts_decay_weight1, 'ts_avg': ts_avg1, 'ts_mid': ts_mid1,
                 'ts_intercept': ts_intercept1,
                 'ts_slope': ts_slope1}
