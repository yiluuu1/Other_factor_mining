import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def split(dfTemp, n):
    for i in range(0, len(dfTemp), n):
        if isinstance(dfTemp, pd.DataFrame):
            yield dfTemp.iloc[i:i + n, :]
        elif isinstance(dfTemp, pd.Series):
            yield dfTemp.iloc[i:i + n]
        else:
            yield dfTemp[i:i + n]


def evaluate(layer_value, hold_period):
    gains = pd.DataFrame()
    for period_value in split(layer_value, hold_period):
        gain = (period_value.iloc[-1, :] - period_value.iloc[0, :])
        gains = pd.concat([gains, gain], axis=1)
    win_ratio = gains.T.apply(lambda x: sum(x >= 0) / len(x))
    win_loss_ratio = gains.T.apply(lambda x: abs(sum(x[x >= 0]) / sum(x[x < 0])))

    def mdd(x):
        return max(1 - x / x.cummax())

    mdd = layer_value.apply(mdd)
    vol = layer_value.diff().std() * np.sqrt(250)
    fv = layer_value.iloc[-1, :]
    mask = fv < 0
    fv[mask] = 1 - fv[mask]
    ret = fv ** (250 / len(layer_value)) - 1
    ret[mask] = - ret[mask]
    sharpe = ret / vol
    eva_res = pd.DataFrame([win_ratio, win_loss_ratio, mdd, ret, vol, sharpe],
                           index=['win_ratio', 'win_loss_ratio', 'mdd', 'ret', 'vol', 'sharpe']).round(3)
    return eva_res


def split_layer(data, hold_period, layer):
    date = data.date.unique()
    layer_value = pd.DataFrame(np.ones((1, layer)), index=['2013-01-01'])

    for period_i in split(date, hold_period):
        period_data = data[(data['date'] >= period_i[0]) & (data['date'] <= period_i[-1])]
        t0_data = data[data.date == period_i[0]]
        q1 = t0_data.factor.quantile(0.8)
        q2 = t0_data.factor.quantile(0.6)
        q3 = t0_data.factor.quantile(0.4)
        q4 = t0_data.factor.quantile(0.2)
        l1_code = t0_data[t0_data.factor > q1].code.values
        l2_code = t0_data[(t0_data.factor <= q1) & (t0_data.factor > q2)].code.values
        l3_code = t0_data[(t0_data.factor <= q2) & (t0_data.factor > q3)].code.values
        l4_code = t0_data[(t0_data.factor <= q3) & (t0_data.factor > q4)].code.values
        l5_code = t0_data[t0_data.factor <= q4].code.values

        l1_value = period_data[period_data.code.isin(l1_code)].groupby('date')['ret'].mean()
        l2_value = period_data[period_data.code.isin(l2_code)].groupby('date')['ret'].mean()
        l3_value = period_data[period_data.code.isin(l3_code)].groupby('date')['ret'].mean()
        l4_value = period_data[period_data.code.isin(l4_code)].groupby('date')['ret'].mean()
        l5_value = period_data[period_data.code.isin(l5_code)].groupby('date')['ret'].mean()
        res = pd.concat([l1_value, l2_value, l3_value, l4_value, l5_value], axis=1, ignore_index=True)
        layer_value = pd.concat([layer_value, res])

    layer_value = layer_value.cumsum().ffill()
    layer_value.columns = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    layer_value.plot()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()
    layer_value['long_short'] = layer_value.iloc[:, 0] - layer_value.iloc[:, -1]+1
    evaluate(layer_value, hold_period).to_csv('eva_res.csv')
