class Trade:
    def __init__(self, allocation=1000000):
        self.trade_price = None
        self.signal = None
        self.pre_signal = 0
        self.commission = 0  # 手续费 0.0002
        self.cash = allocation
        self.position = 0
        self.position_value = 0
        self.value = self.cash
        self.max_price = -0

    def update(self, price, signal):
        self.trade_price = price
        self.signal = signal

    def buy(self):
        self.max_price = self.trade_price
        self.position = self.cash // ((1 + self.commission) * self.trade_price)
        self.position_value = self.position * self.trade_price
        self.cash -= (1 + self.commission) * self.position_value
        self.value = self.cash + self.position_value

    def sell(self):
        self.max_price = -0
        self.position_value = self.position * self.trade_price
        self.cash += (1 - self.commission) * self.position_value
        self.position = 0
        self.position_value = 0
        self.value = self.cash

    def hold(self):
        self.position_value = self.position * self.trade_price
        self.value = self.cash + self.position_value
        if self.trade_price > self.max_price:
            self.max_price = self.trade_price

    def trade(self, loss_stop=False, loss_stop_criteria=1):
        """ 开始交易"""
        if loss_stop and self.position > 0 and self.trade_price <= (1 - loss_stop_criteria) * self.max_price:
            self.signal = 0
        if self.signal == 1 and self.pre_signal == 0 and self.position == 0:
            self.buy()
        elif self.signal == 0 and self.pre_signal == 1 and self.position > 0:
            self.sell()
        elif self.signal == self.pre_signal:
            self.hold()
        else:
            raise ValueError('Invalid "signal" value!')
        self.pre_signal = self.signal
        param_list = ['trade_price', 'signal', 'value', 'position', 'position_value', 'cash']
        trade_info = {name: getattr(self, name) for name in param_list}

        return trade_info
