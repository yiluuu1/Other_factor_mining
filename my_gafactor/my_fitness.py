#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/9 11:04
# @Author : Yilu Jiang
# @File : my_fitness.py
"""Metrics to evaluate the fitness of a program.
"""
import pandas as pd
import numpy as np
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata
# from backtest.Trade import Trade

__all__ = ['make_fitness']


class _Fitness(object):

    def __init__(self, function, greater_is_better, name):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1
        self.name = name

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, name):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic operations into the next generation.

    Parameters
    ----------
    function : callable
        A function that returns a floating point number representative for fitness.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit.

    name: str
        The name of fitness function
    """

    return _Fitness(function=wrap_non_picklable_objects(function), greater_is_better=greater_is_better, name=name)


def _weighted_pearson(y, y_pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred)
        y_demean = y - np.average(y)
        corr = np.sum(y_pred_demean * y_demean) / np.sqrt(np.sum(y_pred_demean ** 2) * np.sum(y_demean ** 2))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred):
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked)


def _mean_absolute_error(y, y_pred):
    return np.average(np.abs(y_pred - y))


def _mean_square_error(y, y_pred):
    return np.average(((y_pred - y) ** 2))


def _root_mean_square_error(y, y_pred):
    return np.sqrt(np.average(((y_pred - y) ** 2)))


def _log_loss(y, y_pred):
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(-score)


def _sharpe_ratio(price, factor):
    comdty = pd.DataFrame({'close': price, 'factor': factor})

    comdty['low_q'] = comdty.factor.rolling(100).quantile(0.2)
    comdty['high_q'] = comdty.factor.rolling(100).quantile(0.8)
    comdty = comdty.dropna().loc[comdty.factor.ne(0).idxmax():, :].reset_index(drop=True)
    if len(comdty) < 200:
        return 0
    conditions = [comdty['factor'].values > comdty['high_q'].values, comdty['factor'].values < comdty['low_q'].values]
    signal = np.select(conditions, [1, -1]).flatten()

    ret = np.log(price).diff().bfill()
    value = (ret*signal).cumsum()+1
    ret = (value[-1] / value[0]) ** (252 / len(value)) - 1
    vol = np.std(np.diff(value) / value[:-1]) * np.sqrt(252)
    if vol == 0:
        return 0
    return ret / vol


def _ir(ret, factor):
    if sum(factor == 0) > 0.3 * len(ret):
        return 0
    # ret = ret[ret.index.isin(factor.index)]
    ic = pd.concat([ret, factor], axis=1).groupby('date', group_keys=False).apply(lambda ft:
                                                                                  ft.iloc[:, 0].corr(ft.iloc[:, 1]))
    ic[np.isinf(ic) | np.isnan(ic)] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ir = abs(ic.mean() / ic.std())
    if np.isnan(ir) or np.isinf(ir):
        return 0
    return ir


def _ic(ret, factor):
    if sum(factor == 0) > 0.3 * len(ret):
        return 0
    ic = ret.corr(factor)
    if np.isinf(ic) or np.isnan(ic):
        return 0
    return ic


weighted_pearson = _Fitness(function=_weighted_pearson, greater_is_better=True, name='pearson')
weighted_spearman = _Fitness(function=_weighted_spearman, greater_is_better=True, name='spearman')
mean_absolute_error = _Fitness(function=_mean_absolute_error, greater_is_better=False, name='mean absolute error')
mean_square_error = _Fitness(function=_mean_square_error, greater_is_better=False, name='mse')
root_mean_square_error = _Fitness(function=_root_mean_square_error, greater_is_better=False, name='rmse')
log_loss = _Fitness(function=_log_loss, greater_is_better=False, name='log loss')
sharpe_ratio = _Fitness(function=_sharpe_ratio, greater_is_better=True, name='sharpe ratio')
ir = _Fitness(function=_ir, greater_is_better=True, name='ir')
ic = _Fitness(function=_ic, greater_is_better=True, name='ic')

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss,
                'sharpe ratio': sharpe_ratio,
                'ir': ir, 'ic': ic}
