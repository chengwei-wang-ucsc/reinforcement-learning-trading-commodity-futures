import gym
from gym import spaces
import pandas as pd
import numpy as np

class TradingEnv_2(gym.Env):
    def __init__(self, obs, price, symbols, initial_fund, both_pos, short_only, long_only, num_stock, commission_rate, valid_folder, margin=0.13, pos_factor=5, mimimum_share=1, slippage=2, 
                commission=0.00015, test=False, valid=False, risk_free=0.028, active_shares=False, leverage=10, cash_out_stop=True, returns=1, calculate_fund=False, cashout_freze=False,
                action_type=1):
        self.action_type = action_type
        self.cashout_freze = cashout_freze
        self.symbols = symbols
        self.margin = margin
        self.leverage = leverage
        self.pos_factor = pos_factor
        self.active_shares = active_shares
        self.num_stock = num_stock
        self.risk_free = risk_free
        self.slippage = slippage
        self.test = test
        if commission_rate is None:
            self.commission = commission
            self.rate = False
        else:
            self.commission = commission_rate
            self.rate = True
        self.mimimum_share = mimimum_share
        self.both_pos = both_pos
        self.short_only = short_only
        self.long_only = long_only
        if self.both_pos == True:
            if self.action_type == 1:
                self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 2: short, 1: long, 0: close
            else:
                self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stock, ), dtype=np.float16)#Number of RowsIns, Numbers Per Row/Number of Actions
            self.trading_option = 'Both Position'
        elif self.long_only == True:
            if self.action_type == 1:
                self.action_space = spaces.Discrete(2) # action is [0, 1] 1: buy, 0: sell
            else:
                self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stock, ), dtype=np.float16)#Number of RowsIns, Numbers Per Row/Number of Actions
            self.trading_option = 'Long Only'
        elif self.short_only == True:
            if self.action_type == 1:
                self.action_space = spaces.Discrete(2) # action is [0, 1] 1: short, 0: close
            else:
                self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stock, ), dtype=np.float16)#Number of RowsIns, Numbers Per Row/Number of Actions
            self.trading_option = 'Short Only'
        self.obs = obs
        self.price = price
        self.obs_index = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs[0].shape))
        self.num_long_pos = 0
        self.num_short_pos = 0
        self.accounts = []
        self.account_1 = []
        self.calculate_fund = calculate_fund
        if self.calculate_fund == True:
            for i in range(self.num_stock):
                try:
                    tmp_array = self.price[:,i].astype('float')
                except:
                    tmp_array = self.price[:,i]
                tmp_fund = np.max(tmp_array)*self.mimimum_share*self.margin*self.leverage*self.pos_factor
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
        else:
            for i in range(self.num_stock):
                tmp_fund = initial_fund
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
        self.open_price = [None]*self.num_stock
        self.held_pos = [None]*self.num_stock
        self.all_his = []
        self.valid = valid
        self.valid_folder = valid_folder
        self.sharpe_his = [self.initial_fund]
        self.cash_his = [self.initial_fund]
        self.unrealized_profit = [0]*self.num_stock
        self.last_symbol = [None]*self.num_stock
        self.cash_out_stop = cash_out_stop
        self.stop_now = False
        self.returns = returns

    def MaxDrawdown(self, return_list):
        try:
            i = np.argmax((np.maximum.accumulate(return_list) - return_list)/np.maximum.accumulate(return_list))
            if i == 0:
                return 0
            j = np.argmax(return_list[:i])
            return (return_list[j] - return_list[i]) / (return_list[j])
        except:
            return -np.inf

    def sharpe_ratio(self, data, risk_free=0.028):
        try:
            data = pd.Series(data)
            data['return'] = (data-data.iloc[0])/data.iloc[0]
            data = data['return'].values
            sharpe = ((np.mean(data)-risk_free)/np.std(data))
        except:
            sharpe = -np.inf
        return sharpe
                     
    def step(self, action):
        obs = self.obs[self.obs_index]
        tmp_price = self.price[self.obs_index]
        tic = self.symbols[self.obs_index]
        try:
            margin = self.margin[self.obs_index]
        except:
            margin = self.margin
        for i in range(self.num_stock):
            tmp_tic = tic[i]
            try:
                tmp_price_1 = tmp_price[i].astype('float')
            except:
                tmp_price_1 = tmp_price[i]
            tmp_action = action#[i]
            tmp_pos = self.held_pos[i]
            open_price = self.open_price[i]
            if self.action_type > 1:
                tmp_action = np.argmax(tmp_action)
            else:
                pass
            if self.short_only == True:
                if tmp_action != 0:
                    tmp_action = 2
                else:
                    pass
            else:
                pass
            if self.active_shares == False:
                tmp_share = self.mimimum_share
            else:
                tmp = int(self.accounts[i]/(tmp_price_1*self.mimimum_share*self.leverage*margin))
                tmp_share = tmp
            if self.cashout_freze==True:
                tmp = int(self.accounts[i]/(tmp_price_1*self.mimimum_share*self.leverage*margin))
                tmp_share = tmp
            else:
                pass

            if tmp_share <= 0 and self.cash_out_stop == True:
                self.stop_now = True
            else:
                pass

            #Contract Change
            if self.last_symbol[i] is None:
                self.last_symbol[i] = tmp_tic
            elif self.last_symbol[i] == tmp_tic:
                self.last_symbol[i] = tmp_tic
            elif self.last_symbol[i] != tmp_tic:
                self.last_symbol[i] = tmp_tic
                try:
                    tmp_price_2 = self.price[self.obs_index-1][i].astype('float64')
                except:
                    tmp_price_2 = self.price[self.obs_index-1][i]
                #Close Long
                if tmp_pos =='long':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = ((tmp_price_2 - self.slippage) - open_price)*tmp_share*self.leverage
                    if self.rate == False:
                        commission = tmp_share*self.commission
                    else:
                        commission = (tmp_price_2 - self.slippage)*tmp_share*self.commission*self.leverage
                    #print('profit_1 Contract Change, close long', profit_1)
                    self.accounts[i] += profit
                    self.account_1[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = None
                    self.unrealized_profit[i] = 0
                    if self.test == True:
                        self.all_his.append(['Function 4-1', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'Contract Change, close long', 
                                            tmp_price_2, tmp_price_2 - self.slippage, self.accounts[i], commission, profit])
                        #Function Index, obs index, loop index, Date, Ticker, action, market price, holding price, cash, commission, profit, accumulate_profit
                    else:
                        pass

                #Close Short
                elif tmp_pos == 'short':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (open_price-(tmp_price_2+self.slippage))*tmp_share*self.leverage
                    if self.rate == False:
                        commission = tmp_share*self.commission
                    else:
                        commission = (tmp_price_2 - self.slippage)*tmp_share*self.commission*self.leverage
                    self.accounts[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] += profit
                    self.account_1[i] -= commission
                    self.held_pos[i] = None
                    self.unrealized_profit[i] = 0                  
                    if self.test == True:
                        self.all_his.append(['Function 4-2', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'Contract Change, close short', 
                                            tmp_price_2, tmp_price_2 + self.slippage, self.accounts[i], commission, profit])
                        #Function Index, obs index, loop index, Date, Ticker, action, market price, holding price, cash, commission, profit, accumulate_profit
                    else:
                        pass
                     
                else:
                    if self.test == True:
                        self.all_his.append(['Function 4-3', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'error', 
                                            tmp_price_2, 0, self.accounts[i], 0, 0])
                        #Function Index, obs index, loop index, Date, Ticker, action, market price, holding price, cash, commission, profit
                    else:
                        pass
            else:
                self.last_symbol[i] = tmp_tic

            #Long
            if tmp_action == 1:
                holding_price = tmp_price_1+self.slippage
                #Open Long
                if tmp_pos == None:
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = tmp_share*self.commission*self.leverage
                    else:
                        commission = self.open_price[i]*tmp_share*self.commission*self.leverage
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'long'
                    self.unrealized_profit[i] = 0
                    self.num_long_pos += 1
                    if self.test == True:
                        self.all_his.append(['Function 1-1', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'open long', 
                                                tmp_price_1, holding_price, self.accounts[i], commission, 0])
                    else:
                        pass

                #Keep Long
                elif tmp_pos == 'long':
                    if self.test == True:
                        self.accounts[i] -= self.unrealized_profit[i]
                        profit = ((tmp_price_1)-open_price)*tmp_share*self.leverage
                        self.accounts[i] += profit
                        self.unrealized_profit[i] = profit
                        self.all_his.append(['Function 1-2', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'hold long', 
                                                tmp_price_1, open_price, self.accounts[i], 0, profit])
                    else:
                        pass
                    
                #Close Short and Open Long
                elif tmp_pos == 'short':
                    ################################Close Short############################################
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (open_price-holding_price)*tmp_share*self.leverage
                    if self.rate == False:
                        commission = tmp_share*self.commission
                    else:
                        commission = (tmp_price_1 - self.slippage)*tmp_share*self.commission*self.leverage
                    self.accounts[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] += profit
                    self.account_1[i] -= commission
                    ################################Open Long##############################################
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = tmp_share*self.commission*self.leverage
                    else:
                        commission = self.open_price[i]*tmp_share*self.commission*self.leverage
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'long'
                    self.unrealized_profit[i] = 0
                    self.num_long_pos += 1
                    if self.test == True:
                        self.all_his.append(['Function 1-3', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'close short and open long', 
                                                tmp_price_1, holding_price, self.accounts[i], commission, profit])
                    else:
                        pass
                    
                else:
                    if self.test == True:
                        self.all_his.append(['Function 1-4', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'error', 
                                                tmp_price_1, 0, self.accounts[i], 0, 0])
                    else:
                        pass
                
            #Short
            elif tmp_action == 2:
                holding_price = tmp_price_1-self.slippage
                #Open Short
                if tmp_pos == None:
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = tmp_share*self.commission*self.leverage
                    else:
                        commission = self.open_price[i]*tmp_share*self.commission*self.leverage
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'short'
                    self.unrealized_profit[i] = 0
                    self.num_short_pos += 1
                    if self.test == True:
                        self.all_his.append(['Function 2-1', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'open short', 
                                                tmp_price_1, holding_price, self.accounts[i], commission, 0])
                    else:
                        pass

                #Keep Short
                elif tmp_pos == 'short':
                    if self.test == True:
                        self.accounts[i] -= self.unrealized_profit[i]
                        profit = (open_price-(tmp_price_1))*tmp_share*self.leverage
                        self.accounts[i] += profit
                        self.unrealized_profit[i] = profit
                        self.all_his.append(['Function 2-2', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'hold short', 
                                                tmp_price_1, open_price, self.accounts[i], 0, profit])
                    else:
                        pass
                    
                #Close Long and Open Short
                elif tmp_pos == 'long':
                    ################################Close Long############################################
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (holding_price - open_price)*tmp_share*self.leverage
                    if self.rate == False:
                        commission = tmp_share*self.commission
                    else:
                        commission = (tmp_price_1 - self.slippage)*tmp_share*self.commission*self.leverage
                    self.accounts[i] += profit
                    self.account_1[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    ################################Open Short##############################################
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = tmp_share*self.commission*self.leverage
                    else:
                        commission = self.open_price[i]*tmp_share*self.commission*self.leverage
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'short'
                    self.unrealized_profit[i] = 0
                    self.num_short_pos += 1
                    if self.test == True:
                        self.all_his.append(['Function 2-3', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'close long and open short', 
                                                tmp_price_1, holding_price, self.accounts[i], commission, profit])
                    else:
                        pass
                    
                else:
                    if self.test == True:
                        self.all_his.append(['Function 2-4', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'error', 
                                                tmp_price_1, 0, self.accounts[i], 0, 0])
                    else:
                        pass

            #Close
            if tmp_action == 0:
                #Close Long
                if tmp_pos =='long':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = ((tmp_price_1 - self.slippage) - open_price)*tmp_share*self.leverage
                    if self.rate == False:
                        commission = tmp_share*self.commission
                    else:
                        commission = (tmp_price_1 - self.slippage)*tmp_share*self.commission*self.leverage
                    self.accounts[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] += profit
                    self.account_1[i] -= commission
                    self.held_pos[i] = None
                    self.unrealized_profit[i] = 0
                    if self.test == True:
                        self.all_his.append(['Function 3-1', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'close long', 
                                                tmp_price_1, tmp_price_1 - self.slippage, self.accounts[i], commission, profit])
                    else:
                        pass

                #Close Short
                elif tmp_pos == 'short':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (open_price - (tmp_price_1 + self.slippage))*tmp_share*self.leverage
                    if self.rate == False:
                        commission = tmp_share*self.commission
                    else:
                        commission = (tmp_price_1 + self.slippage)*tmp_share*self.commission*self.leverage
                    self.accounts[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] += profit
                    self.account_1[i] -= commission
                    self.held_pos[i] = None 
                    self.unrealized_profit[i] = 0
                    if self.test == True:
                        self.all_his.append(['Function 3-2', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'close short', 
                                                tmp_price_1, tmp_price_1 + self.slippage, self.accounts[i], commission, profit])
                    else:
                        pass
                     
                else:
                    if self.test == True:
                        self.all_his.append(['Function 3-3', self.held_pos[i], tmp_action, self.obs_index, i, tmp_share, tmp_price[-1], tmp_tic, 'error', 
                                                tmp_price_1, 0, self.accounts[i], 0, 0])
                    else:
                        pass

        self.cash_his.append(np.sum(self.accounts))
        self.sharpe_his.append(np.sum(self.account_1))
        if self.returns == 1:
            returns = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
            maxdrawn = self.MaxDrawdown(self.cash_his)
            reward = returns - abs(maxdrawn)
        elif self.returns == 2:
            reward = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
        elif self.returns == 3:
            returns = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
            maxdrawn = self.MaxDrawdown(self.cash_his)
            sharpe = self.sharpe_ratio(self.sharpe_his, self.risk_free)
            reward = ((returns-maxdrawn)*0.5)+(sharpe*0.5)
        else:
            pass

        if self.obs_index == self.obs.shape[0]-1 or self.stop_now == True:
            done = True
            sharpe = self.sharpe_ratio(self.sharpe_his, self.risk_free)
            maxdrawn = self.MaxDrawdown(self.cash_his)
            return_rate = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
            print('------------------This Round Info------------------')
            print('Active Shares: ', self.active_shares, ', Trading Option: ', self.trading_option)
            print('Long Trades: ', self.num_long_pos, ', Short Trades: ', self.num_short_pos)
            print('Step Index: ', self.obs_index)
            print('Initial Account Balance: ', self.initial_fund)
            print('End Account Balance: ', self.cash_his[-1])
            print('Rate of Return: ', return_rate)
            print('Sharpe Ratio: ', sharpe)
            print('MaxdrawDown: ', maxdrawn)
            print('------------------This Round Ends------------------')
            info = {'done': True, 'return rate':return_rate, 'maxdrawndown': maxdrawn, 'sharpe': sharpe, 'account_history':self.cash_his}
        else:
            done = False   
            info = {'done':False, 'return rate':None, 'maxdrawndown': None, 'sharpe': None, 'account_history':None}

        if self.valid==True and done==True:
            file_name = str(self.valid_folder+'backtest_'+str(reward)+'_.npy')
            self.cash_history = np.asarray(self.cash_his)
            np.save(file_name, self.cash_his)
            if self.test == True:
                test_df = pd.DataFrame(self.all_his)
                test_df.columns = ['Function Index', 'current holdings', 'model action', 'obs index', 'Loop Index', 'trading shares', 'Date', 'Ticker', 'action', 'market price', 
                                    'holding price', 'cash', 'commission', 'profit']
                test_df.to_csv(self.valid_folder+'test_df.csv')
        else:
            pass   
        self.obs_index += 1
        return obs, reward, done, info

    def reset(self):
        self.obs_index = 0
        self.num_long_pos = 0
        self.num_short_pos = 0
        self.accounts = []
        self.account_1 = []
        if self.calculate_fund == True:
            for i in range(self.num_stock):
                try:
                    tmp_array = self.price[:,i].astype('float')
                except:
                    tmp_array = self.price[:,i]
                tmp_fund = np.max(tmp_array)*self.mimimum_share*np.max(self.margin)*self.leverage*self.pos_factor
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
        else:
            for i in range(self.num_stock):
                tmp_fund = self.initial_fund
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
        self.initial_fund = np.sum(self.accounts)
        self.open_price = [None]*self.num_stock
        self.held_pos = [None]*self.num_stock
        self.all_his = []
        self.sharpe_his = [self.initial_fund]
        self.cash_his = [self.initial_fund]
        self.unrealized_profit = [0]*self.num_stock
        obs = self.obs[self.obs_index]
        self.last_symbol = [None]*self.num_stock
        self.stop_now = False
        return obs