#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/9 11:05
# @Author : Yilu Jiang
# @File : my_utils.py

import numbers
import numpy as np
from joblib import cpu_count

unit_map = {'weight': {'吨', '磅', '克', '斤', 'g', '重量箱'}, 'money': {'元', '英镑', '欧元'},
            'area': {'顷', '平方', '亩'},
            'pct': {'百分比', '%', '*', None}, 'time': {'天', '周', '月', '日'},
            'volume': {'桶', '包', '升', '加仑', '立方'}}


def unit_transform(unit_dict):
    res = unit_dict.copy()
    for index, unit in unit_dict.items():
        weight = len(unit_map['weight'].intersection(set(unit)))
        volume = len(unit_map['volume'].intersection(set(unit)))
        money = len(unit_map['money'].intersection(set(unit)))
        time = len(unit_map['time'].intersection(set(unit)))
        area = len(unit_map['area'].intersection(set(unit)))
        pct = unit in unit_map['pct']
        if '/' in unit:
            if money > 0:
                res[index] = 'money/' + weight * 'weight' + volume * 'volume'
            elif area + time > 0:
                res[index] = weight * 'weight' + volume * 'volume' + '/' + time * 'time' + area * 'area'
            else:
                pass
        else:
            if pct:
                res[index] = None
            elif weight + volume + area + time + money > 0:
                res[index] = weight * 'weight' + volume * 'volume' + money * 'money' + time * 'time' + area * 'area'
            else:
                pass
    return res


def check_unit(function, terminals, *args):
    try:
        if function in ('cs_rank', 'ts_rank', 'argmax', 'argmin', 'ts_corr'):
            return None
        elif function in ('add', 'sub'):
            if terminals[0] == terminals[1]:
                return terminals[0]
            elif terminals[0] is None:
                return terminals[1]
            elif terminals[1] is None:
                return terminals[0]
            else:
                return 'wrong'
        elif function == 'mul':
            if terminals[0] is None:
                return terminals[1]
            elif terminals[1] is None:
                return terminals[0]
            elif ('/' + terminals[0]) in terminals[1]:
                return terminals[1].replace('/' + terminals[0], '', 1)
            elif ('/' + terminals[1]) in terminals[0]:
                return terminals[0].replace('/' + terminals[1], '', 1)
            else:
                return terminals[0] + '*' + terminals[1]
        elif function == 'div':
            if terminals[0] is None:
                if terminals[1] is None:
                    return terminals[1]
                else:
                    return '/' + terminals[1]
            elif terminals[1] is None:
                return terminals[0]
            if ('*' + terminals[0]) in terminals[1]:
                return '/' + terminals[1].replace('*' + terminals[0], '', 1)
            elif ('*' + terminals[1]) in terminals[0]:
                return terminals[0].replace('*' + terminals[1], '', 1)
            if terminals[0] == terminals[1]:
                return None
            else:
                return terminals[0] + '/' + terminals[1]
        elif function == 'inv':
            if terminals[0] is None:
                return None
            if '/' in terminals[0]:
                temp = terminals[0].split('/')
                if len(temp[0]) == 0:
                    return temp[1]
                else:
                    return temp[1] + '/' + temp[0]
            return '/' + terminals[0]
        elif function == 'ts_cov':
            if terminals[0] is None:
                return terminals[1]
            elif terminals[1] is None:
                return terminals[0]
            else:
                return terminals[0] + '*' + terminals[1]
        elif function == 'power':
            return terminals + ('*' + terminals) * (args[0] - 1)
        else:
            return terminals[0]
    except:
        print(function, terminals)


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)


def _partition_estimators(n_estimators, n_jobs):
    """Compute the number of jobs."""
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
        if np.isnan(ft).all() or len(ft)==1:
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
