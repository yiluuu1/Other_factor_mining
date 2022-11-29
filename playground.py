import pandas as pd
import numpy as np
from my_gplearn.my_genetic import SymbolicRegressor
from factor_test.split_layer import split_layer
import matplotlib.pyplot as plt

data = pd.read_csv('oral.csv', encoding='GBK')
data.set_index('date', inplace=True)
data.index = pd.to_datetime(data.index)
data['tmr_ret'] = np.log(data.close).diff().bfill()
data = pd.concat([data], keys=['oral'], names=['code']).swaplevel()
unit_dict = pd.read_excel('oral_unit.xls').to_dict('records')[0]

gp = SymbolicRegressor(metric='ic', population_size=100, generations=10, tournament_size=20,
                       stopping_criteria=1, const_range=None, init_depth=(1, 4), p_crossover=0.8,
                       p_subtree_mutation=0.01, p_hoist_mutation=0.1, p_point_mutation=0.01, n_jobs=16, verbose=1,
                       parsimony_coefficient=0.002, feature_names=data.columns[:-1], unit_dict=unit_dict,
                       function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'inv', 'power', 'cs_rank',
                                     'scale', 'ts_delay', 'ts_delta', 'ts_corr', 'ts_rank', 'ts_sum', # 'ts_cov',
                                     'ts_prod', 'ts_std', 'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin',
                                     'ts_decay_linear', 'ts_avg', 'ts_mid', 'ts_intercept', 'ts_slope', 'neg'))
gp.fit(data.iloc[:, :-1], data.tmr_ret)
print(gp)

data['factor'] = gp.predict(data.iloc[:, :-1])
