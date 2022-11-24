import numpy as np
import pandas as pd
from pandas.core import series
from pandas.core.frame import DataFrame


class Evaluate:
    def __init__(self, trade_data) -> None:
        self.trade_data = trade_data

    def cal_max_drawdown(self, data):
        # supply_demand_data 最好以时间为 index
        cummax_price = data.cummax()
        drawdown = (data - cummax_price) / cummax_price  # 每天的回撤率
        return drawdown.min()

    # 获取最大回撤，最大回撤时间
    def get_max_drawdown_data(self) -> DataFrame:
        price_data = self.trade_data.copy()
        # 年度   最大回撤 & 最大回撤时间
        max_dd_data = price_data.value.resample('y').apply(self.cal_max_drawdown).apply(pd.Series)
        # 最后一个apply，将 tuple series转变为两列的dataframe
        max_dd_data.columns = ['Maximum Drawdown']
        return max_dd_data

    # 只有 0-1仓位时获取
    def get_holding_gain(self) -> DataFrame:
        # 开平仓收益---> 后续来计算盈亏比 & 胜率
        condition1 = (self.trade_data['signal'] != 0)
        yesterday=self.trade_data['signal'].shift(1)
        condition2 = (yesterday == 0) | (self.trade_data.index == self.trade_data.index[0])
        buy_trade = self.trade_data[condition1 & condition2]['value'].reset_index()

        condition1 = (yesterday == 1) | (yesterday == -1)
        condition2 = (self.trade_data['signal'] == 0) | (self.trade_data.index == self.trade_data.index[-1])
        sell_trade = self.trade_data[condition1 & condition2]['value'].reset_index()

        # 计算开平仓结果数据
        buy_trade['gain'] = sell_trade['value'] - buy_trade['value']
        buy_trade['hold_day'] = (sell_trade['index'] - buy_trade['index']).dt.days
        self.holding_data = buy_trade.copy().set_index('index')
        return self.holding_data

    def get_holding_perform(self) -> DataFrame:
        # 计算每次开平仓的收益 --> 算胜率、盈亏比
        self.get_holding_gain()  # 计算开平仓收益

        win_ratio = self.holding_data.gain.resample("y").apply(lambda x: x[x > 0].count() / x.count())
        avg_hold_day = self.holding_data.hold_day.resample("y").apply(lambda x: np.average(x))

        def cal_win_loss_ratio(x):
            if np.isnan(x[x < 0].mean()):
                return 'No loss'
            else:
                return x[x >= 0].mean() / abs(x[x < 0].mean())

        win_loss_ratio = self.holding_data.gain.resample("y").apply(cal_win_loss_ratio)

        year_hold_perform = pd.concat([win_ratio, win_loss_ratio, avg_hold_day], axis=1,
                                      keys=['win_ratio', 'win_loss_ratio', 'avg_hold_days'])
        return year_hold_perform

    def get_sharpe(self, ret_vol_data) -> series:
        # 夏普比率： 策略平均收益率/波动率
        sharpe = np.divide(ret_vol_data['Annualized Return'], ret_vol_data['Annualized Volatility'])
        sharpe.name = 'Sharpe Ratio'
        return sharpe

    def get_volatility(self) -> series:
        # 日度数据 计算年度波动率
        ret_data = self.trade_data['value'].pct_change()
        stra_vol = ret_data.resample("y").agg(np.std) * np.sqrt(252)  # 日度收益率算出来的波动率-->年化
        stra_vol.name = 'Annualized Volatility'
        return stra_vol

    def get_ret_vol(self):
        ret = self.trade_data['value'].resample('y').apply(lambda x: (x[-1] / x[0]) ** (252 / len(x)) - 1)
        ret.name = 'Annualized Return'
        vol = self.get_volatility()
        ret_vol_data = pd.concat([ret, vol], axis=1)
        return ret_vol_data

    def evaluate(self):
        # 获取持仓表现： 胜率、盈亏比
        holding_perform = self.get_holding_perform()

        # 计算 收益率、波动率、夏普比率
        ret_vol_strategy = self.get_ret_vol()
        sharpe = self.get_sharpe(ret_vol_strategy)

        perform_data = pd.concat([holding_perform, ret_vol_strategy, sharpe], axis=1)

        max_drawdown = self.get_max_drawdown_data()
        self.evaluate_data = pd.merge(max_drawdown, perform_data, left_index=True, right_index=True, how='left')
        self.evaluate_data.index = self.evaluate_data.index.year
        self.evaluate_data.index.name = 'Year'
        self.evaluate_data = self.evaluate_data.round(3)
        return self.evaluate_data
