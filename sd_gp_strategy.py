import pandas as pd
import numpy as np
from backtest.Trade import Trade
from backtest.Evaluate import Evaluate
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from my_gafactor.my_genetic import *


def recom_indicator(data):
    corr_res = data.groupby(data.date.apply(lambda x: x.year)).corr(numeric_only=True)
    corr_res = corr_res[corr_res.index.get_level_values(0) >= 2017]

    corr_res = corr_res.groupby(level=1).median().fillna(0)
    corr_res = corr_res.reindex(corr_res.close.abs().sort_values(ascending=False).index)
    corr_res = corr_res[corr_res.close.abs() >= 0.15]

    check = corr_res.iloc[1:, 1:]
    for i in range(len(check)):
        try:
            a = check.loc[:, check.index[i]]
            check = check[(a.abs() < 0.5) | (a == 1)]
        except:
            break

    corr_res = corr_res[check.index].T.iloc[:, 0]
    return corr_res.index.values


def strategy(comdty, mode):
    file_dir = 'gp_results/' + code + '/' + mode

    comdty['low_q'] = comdty.factor.rolling(100).quantile(0.2)
    comdty['high_q'] = comdty.factor.rolling(100).quantile(0.8)
    comdty = comdty.dropna().loc[comdty.factor.ne(0).idxmax():, :]
    conditions = [comdty['factor'].values > comdty['high_q'].values, comdty['factor'].values < comdty['low_q'].values]
    signal = np.select(conditions, [1, -1]).flatten()

    backtest_dt = comdty.date.values.flatten()
    market_inpulse = 0  # np.random.randint(-5, 6)*0.5

    long_signal = dict(zip(backtest_dt, np.where(signal < 0, 0, signal)))
    short_signal = dict(zip(backtest_dt, -np.where(signal > 0, 0, signal)))
    trade_price = dict(zip(backtest_dt, comdty.close.values.flatten()))
    l_trade_dict, s_trade_dict = {}, {}
    long_trade = Trade()
    short_trade = Trade()
    for date in backtest_dt:
        long_trade.update(price=trade_price[date] + market_inpulse, signal=long_signal[date])
        l_trade_dict[date] = long_trade.trade(loss_stop=True, loss_stop_criteria=0.05)
        short_trade.update(price=trade_price[date] + market_inpulse, signal=short_signal[date])
        s_trade_dict[date] = short_trade.trade(loss_stop=True, loss_stop_criteria=0.05)
    long_data = pd.DataFrame.from_dict(l_trade_dict, 'index')
    short_data = pd.DataFrame.from_dict(s_trade_dict, 'index')
    long_data.value = long_data.value / long_data.value[0]
    short_data.value = 2 - short_data.value / short_data.value[0]
    value_data = pd.DataFrame(long_data.value * short_data.value)
    value_data['signal'] = signal
    value_data['price'] = comdty.close.values
    evaluate_res = Evaluate(value_data).evaluate()

    value_data.to_csv(file_dir + '/signal.csv')
    evaluate_res.to_csv(file_dir + '/evaluate_data.csv')

    value_data['ret'] = value_data['value'].pct_change()
    value_data.loc[:, 'cummax_price'] = value_data['value'].cummax()
    value_data.loc[:, 'dd'] = -np.subtract(value_data.loc[:, 'cummax_price'], value_data.loc[:, 'value'])
    value_data.loc[:, 'dd_ratio'] = np.divide(value_data.loc[:, 'dd'], value_data.loc[:, 'cummax_price'])
    value_data.index = value_data.index.astype(str)
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(6, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :])
    plt.bar(value_data.index, height=value_data['dd_ratio'])
    plt.ylabel("Rolling Drawdown Ratio")
    plt.xticks([])
    ax2 = fig.add_subplot(gs[2:, 0])
    plt.plot(value_data['value'], 'k')
    plt.ylabel('Strategy Value')
    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(20)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(8))
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(file_dir + '/value_with_drawdown.png')
    plt.show()
    plt.close()


code = '生猪'
comdty = pd.read_csv('data/supply_demand_data/' + code + '.csv', index_col=[0,1])
comdty = comdty[comdty.close.notna()].ffill()
test_period = 200

train_close = comdty.iloc[:, :].close  #
test_close = comdty.iloc[-test_period - 99:, :].close
train_date = comdty.iloc[:-test_period, :].date
test_date = comdty.iloc[-test_period - 99:, :].date
comdty.close = np.log(comdty.close).shift(-1)
# train_indicators = comdty.loc[comdty.index[:-test_period], recom_indicator(comdty)].fillna(0) #
# test_indicators = comdty.loc[comdty.index[-test_period - 99:], recom_indicator(comdty)].fillna(0)
train_indicators = comdty.iloc[:, 2:].fillna(0)  #
test_indicators = comdty.iloc[-test_period - 99:, 2:].fillna(0)
gp = SymbolicRegressor(metric='sharpe ratio', population_size=900, generations=10, tournament_size=20,
                       stopping_criteria=10, const_range=(-1., 1.), init_depth=(1, 4), p_crossover=0.8,
                       p_subtree_mutation=0.01, p_hoist_mutation=0.1, p_point_mutation=0.01, n_jobs=8, verbose=1,
                       parsimony_coefficient=0.2,
                       function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'inv', 'power', 'cs_rank',
                                     'scale', 'ts_delay', 'ts_delta', 'ts_corr', 'ts_cov', 'ts_rank', 'ts_sum',
                                     'ts_prod', 'ts_std', 'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin',
                                     'ts_decay_linear', 'ts_avg', 'ts_mid', 'ts_intercept', 'ts_slope', 'neg'))

gp.fit(train_indicators, train_close)
x = gp.__str__()
open('gp_results/' + code + '/factor.txt', 'w').write(x)
print(x)

factor = gp.predict(train_indicators)
train_comdty = pd.DataFrame({'date': train_date, 'close': train_close, 'factor': factor})
strategy(train_comdty, mode='train')

factor = gp.predict(test_indicators)
test_comdty = pd.DataFrame({'date': test_date, 'close': test_close, 'factor': factor})
strategy(test_comdty, mode='test')

print('done')
