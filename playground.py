import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

lv = pd.read_csv('data/supply_demand_data/焦煤.csv').ffill().fillna(0)
lv.set_index('date', inplace=True)
lv =lv.loc[:, ['样本洗煤厂（110家）：精煤：库存：中国（周）', 'close']]

lv['factor'] = lv.iloc[:, 0]
lv =lv[~np.isinf(lv.factor) & ~np.isnan(lv.factor)]
lv = lv.loc[max(lv.factor.ne(0).idxmax(), lv.close.ne(0).idxmax()):, :]
#lv = (lv-lv.mean())/lv.std()

lv.loc[lv.index[100:],['factor', 'close']].plot(secondary_y='factor')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
