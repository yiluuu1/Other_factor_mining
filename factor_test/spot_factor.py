import pandas as pd
import numpy as np
from my_gplearn.my_genetic import SymbolicRegressor
from factor_test.split_layer import split_layer
import matplotlib.pyplot as plt



spot = pd.read_excel('../data/spot_data/spot_data.xlsx').ffill()
spot.set_index('date', inplace=True)
spot.index = pd.to_datetime(spot.index)
futures = pd.read_excel('../data/spot_data/all_close.xlsx').ffill()
futures.set_index('date', inplace=True)
ret = np.log(futures).diff().fillna(method='bfill', limit=1)
tmr_ret = ret.shift(-20)
futures = futures[spot.columns][futures.index < '20210101']
ret = ret[spot.columns][ret.index < '20210101']
tmr_ret = tmr_ret[spot.columns][tmr_ret.index < '20210101']

train_data = pd.DataFrame({'spot': spot.stack(), 'ret': ret.stack(),
                           'futures': futures.stack(), 'tmr_ret': tmr_ret.stack()})
unit_dict = {'spot': '元/吨', 'ret': '百分比', 'futures': '元/吨', 'tmr_ret': '百分比'}
train_data.index = train_data.index.set_names(['date', 'code'])
gp = SymbolicRegressor(metric='ir', population_size=100, generations=10, tournament_size=20,
                       stopping_criteria=1, const_range=None, init_depth=(1, 4), p_crossover=0.8,
                       p_subtree_mutation=0.01, p_hoist_mutation=0.1, p_point_mutation=0.01, n_jobs=16, verbose=1,
                       parsimony_coefficient=0.002, feature_names=train_data.columns[:-1], unit_dict=unit_dict,
                       function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'inv', 'power', 'cs_rank',
                                     'scale', 'ts_delay', 'ts_delta', 'ts_corr', 'ts_cov', 'ts_rank', 'ts_sum',
                                     'ts_prod', 'ts_std', 'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin',
                                     'ts_decay_linear', 'ts_avg', 'ts_mid', 'ts_intercept', 'ts_slope', 'neg'))
gp.fit(train_data.iloc[:, :-1], train_data.tmr_ret)
print(gp)

train_data['factor'] = gp.predict(train_data.iloc[:, :-1])
# train_data = train_data.dropna()
ic = train_data[['factor', 'tmr_ret']].groupby('date', group_keys=False).apply(
    lambda ft: ft.iloc[:, 0].corr(ft.iloc[:, 1]))
ir = ic.mean() / ic.std()
if ir < 0:
    print('short factor')
    train_data['factor'] = -train_data['factor']
    ic = train_data[['factor', 'tmr_ret']].groupby('date', group_keys=False).apply(
        lambda ft: ft.iloc[:, 0].corr(ft.iloc[:, 1]))
    ir = ic.mean() / ic.std()
ic_pct = sum(ic >= 0) / len(ic)
ic_res = pd.DataFrame([[ir, ic.mean(), ic.std(), ic_pct]], columns=['ir', 'ic_mean', 'ic_std', 'ic_pct']).round(6)
print(ic_res)
ic.cumsum().plot(color='k')
plt.title(gp)
plt.legend(labels=['ic_cumsum'])
plt.grid(True)
plt.tight_layout()
plt.show()

train_data = train_data.reset_index()
train_data['date'] = train_data['date'].apply(pd.Timestamp.strftime, args=('%Y-%m-%d',))
split_layer(train_data[['date', 'code', 'factor', 'ret']], 20, 5)
