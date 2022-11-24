import numpy as np
import pandas as pd


# 附数据实例，不能只输入现货与期货价格list，还需要知道价格对应的日期与品种才可以,并将其设置为multiindex

def preprocess(x):
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


def ts_argmax(x1, window):
    return x1.groupby(level='code', group_keys=False).rolling(window).apply(
        np.argmax, raw=True, engine='numba').droplevel(0)


def ts_rank(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).rank(pct=True).droplevel(0)


def div(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    res = x1 / x2
    res[np.isinf(res)] = (res[~np.isinf(res)]).mean()
    return res


def ts_avg(x1, window):
    return x1.groupby('code', group_keys=False).rolling(window).mean().droplevel(0)


def ts_intercept(x1, window):
    def intercept(ft):
        x = np.arange(window) + 1
        beta = np.sum((x - np.mean(x)) * (ft - np.mean(ft))) / np.sum((x - np.mean(x)) ** 2)
        return np.mean(ft) - beta * np.mean(x)

    return x1.groupby('code').rolling(window).apply(intercept, raw=True, engine='numba').droplevel(0)


def ts_slope(x1, window):
    def slope(ft):
        x = np.arange(window) + 1
        return np.sum((x - np.mean(x)) * (ft - np.mean(ft))) / np.sum((x - np.mean(x)) ** 2)

    return x1.groupby('code').rolling(window).apply(slope, raw=True, engine='numba').droplevel(0)


def cs_rank(x1):
    return x1.groupby('date', group_keys=False).rank(pct=True)


def ts_delta(x1, window):
    return x1.groupby('code', group_keys=False).diff(window)


def factor1(data):
    return cs_rank(div(data.spot, data.futures))


def factor2(data):
    return ts_argmax(ts_argmax(data.spot, 11), 11) + ts_argmax(data.spot, 11)


def factor3(data):
    return data.spot - data.futures


def factor4(data):
    return ts_rank(ts_rank(data.spot, 20), 34)


def factor5(data):
    return -ts_intercept(cs_rank(ts_delta(data.spot, 1)), 23)


def factor6(data):
    return -ts_avg(ts_avg(ts_slope(abs(data.futures), 41), 34), 34)


sample = pd.read_csv('sample_data.csv', encoding='GBK', index_col=[0, 1])
f1 = factor1(sample)
f2 = factor2(sample)
f3 = factor3(sample)
f4 = factor4(sample)
f5 = factor5(sample)
f6 = factor6(sample)

all_fac = pd.concat([f1, f2, f3, f4, f5, f6], axis=1)
all_fac.columns = [1, 2, 3, 4, 5, 6]
all_fac.corr()
