#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/9 11:05
# @Author : Yilu Jiang
# @File : my_utils.py
"""
Most of these functions are slightly modified versions of some key utility
functions from scikit-learn that gplearn depends upon. They reside here in
order to maintain compatibility across different versions of scikit-learn.

"""

import numbers
import numpy as np
from joblib import cpu_count

def unit_transform(unit_dict):
    res = unit_dict.copy()
    for index, unit in unit_dict:
        if set('吨' '磅' '克' '盎司' '斤' 'g').intersection(set(unit)):
            res[index] = 'weight'
        elif set('美元' '元' '英镑' '欧元').intersection(set(unit)):
            res[index] = 'money'
        if set('天' '周' '月').intersection(set(unit)):
            res[index] = 'time'
        if set('公顷' 'm^2').intersection(set(unit)):
            res[index] = 'area'
        if set('桶' '包' '公升' '加仑').intersection(set(unit)):
            res[index] = 'volume'
        else:
            pass
    return res

def check_unit(function, terminals, *args):
    if function in ('cs_rank', 'ts_rank', 'argmax', 'argmin', 'ts_corr'):
        return None
    elif function in ('add', 'sub'):
        if terminals[0] == terminals[1]:
            return terminals[0]
        else:
            return False
    elif function == 'mul':
        if ('*' + terminals[0]) in terminals[1]:
            return terminals[1] - ('*' + terminals[0])
        elif ('*' + terminals[1]) in terminals[0]:
            return terminals[0] - ('*' + terminals[1])
        else:
            return terminals[0] + '*' + terminals[1]
    elif function == 'div':
        if ('/' + terminals[0]) in terminals[1]:
            return terminals[1] - ('/' + terminals[0])
        elif ('/' + terminals[1]) in terminals[0]:
            return terminals[0] - ('/' + terminals[1])
        else:
            return terminals[0] + '/' + terminals[1]
    elif function == 'inv':
        return '/' + terminals
    elif function == 'ts_cov':
        return terminals[0] + '*' + terminals[1]
    elif function == 'power':
        return terminals + ('*' + terminals) * (args[0] - 1)
    else:
        return terminals

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise, raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = cpu_count() + 1 + n_jobs

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs, dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def preprocess(x):
    # res = x.copy().dropna()
    # for date, ft in res.groupby('date', group_keys=False):
    #     fm = ft.median()
    #     fm1 = abs(ft - fm).median()
    #     ft[ft > (fm + 5 * fm1)] = fm + 5 * fm1
    #     ft[ft < (fm - 5 * fm1)] = fm - 5 * fm1
    #     ft = (ft - ft.mean()) / ft.std()
    #     res[date] = ft
    # return res
    res = x.copy()
    for date, ft in res.groupby('date', group_keys=False):
        if np.isnan(ft).all():
            pass
        else:
            fm = ft.median()
            fm1 = abs(ft - fm).median()
            ft[ft > (fm + 5 * fm1)] = fm + 5 * fm1
            ft[ft < (fm - 5 * fm1)] = fm - 5 * fm1
            ft = (ft - ft.mean()) / ft.std()
        ft = ft.fillna(0)
        res[date] = ft
    return res
