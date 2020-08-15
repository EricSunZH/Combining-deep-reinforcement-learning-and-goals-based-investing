import gym
import gym.spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from function import *
import csv
from settings import *
import warnings
warnings.filterwarnings('ignore')


class PortfolioEnv(gym.Env):
    """
    Reference: (1) https://stable-baselines.readthedocs.io/en/master/index.html
               (2) https://github.com/wassname/rl-portfolio-management
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 steps,  
                 trading_cost=0.0001,
                 time_cost=0.00,
                 window_length=3,
                 start_idx=0,
                 sample_start_date=None   
                ):

        assert ADD_INVEST.shape[0] == steps + 1, "ADD_INVEST and defined steps must have the same shape"
        datafile = DATA_DIR               
        history, abbreviation = read_stock_history(filepath=datafile)
        self.history = history              
        self.abbreviation = abbreviation    
        self.trading_cost = trading_cost
        self.window_length = window_length  
        self.num_stocks = history.shape[0]  
        self.start_idx = start_idx
        self.csv_file = CSV_DIR             

        self.src = DataGenerator(history=history,
                                 abbreviation=abbreviation,
                                 steps=steps,
                                 window_length=window_length,
                                 start_idx=start_idx,
                                 start_date=sample_start_date)

        self.sim = PortfolioSim(asset_names=abbreviation,
                                trading_cost=trading_cost,
                                time_cost=time_cost,
                                steps=steps)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(len(abbreviation), window_length, history.shape[-1]-1), 
                                                dtype=np.float32)

        self.action_space = gym.spaces.Box(low=0, high=1, 
                                           shape=(len(self.src.asset_names)+1, ), 
                                           dtype=np.float32)


    def step(self, action):
        return self._step(action)

    def _step(self, action):
        np.testing.assert_almost_equal(action.shape, (len(self.sim.asset_names) + 1,))

        weights = np.clip(action, 0, 1)
        weights /= (weights.sum() + EPS)
        weights[0] += np.clip(1 - weights.sum(), 0, 1) 

        # Sanity checks
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        observation, done1, ground_truth_obs, obs_ = self.src._step() 

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length+1, observation.shape[2]))
        observation_concat = np.concatenate((cash_observation, obs_), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        y1 = observation_concat[:, -1, 3] / observation_concat[:, -2, 3]
        
        reward, info, done2 = self.sim._step(weights, y1, self.src.step)

        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs

        self.infos.append(info)

        if done1:
            keys = self.infos[0].keys()
            with open(self.csv_file, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)
        return observation, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        info = {}
        return observation
            
    def render(self, mode='human', close=False):
        plt.figure(figsize=(10,5))
        df_info = pd.read_csv(CSV_DIR)
        df_info = market_value(df_info, self.trading_cost)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        
        # Calculate total weight with constraint of sum of each weight = 1
        df_sum = []
        for i in range(df_info.shape[0]):
            sum_ = df_info.iloc[i, 5:(5+len(self.abbreviation)+1)].sum()
            df_sum.append(sum_)

        self.df_info = df_info
        self.mdd = mdd
        self.sharpe_ratio = sharpe_ratio
        self.df_sum = df_sum

        self.worth_ax =plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
        self._render_price()
        self.weight_ax = plt.subplot2grid((2, 1), (1, 0), rowspan=1, colspan=1)
        self._render_weight()
        plt.suptitle(
                "GOAL={: }".format(GOAL)+ "  " +
                "sharpe_ratio={: 2.4f}".format(self.sharpe_ratio)
        )
        plt.setp(self.worth_ax.get_xticklabels(), visible=False)
        plt.xlabel('Time Steps')
        plt.show(block=False)

        plt.pause(0.001)

    def _render_price(self):   
        plt.figure(figsize=(10,5))
        self.worth_ax.clear()
        self.worth_ax.plot(self.df_info["portfolio_value"], label='portfolio')
        self.worth_ax.plot(self.df_info["benchmark"], label='benchmark')
        last_date = self.df_info.index[-1]
        last_worth = self.df_info['portfolio_value'].values[-1]
        last_market = self.df_info['market_value'].values[-1]
        self.worth_ax.annotate('{0:.2f}'.format(last_worth),     
                               (last_date, last_worth),
                               xytext=(last_date, last_worth),
                               bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")
        self.worth_ax.annotate('{0:.2f}'.format(last_market),     
                               (last_date, last_market),
                               xytext=(last_date, last_market),
                               bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")        
        self.worth_ax.set_title("Total Wealth")
        self.worth_ax.legend()

    def _render_weight(self):
        self.weight_ax.clear()
        self.weight_ax.plot(self.df_info['weights1'],  label='Cash', alpha=0.5)
        self.weight_ax.plot(self.df_info['weights2'],  label='Russel 1000 Growth ETF', alpha=0.5)
        self.weight_ax.plot(self.df_info['weights3'],  label='US Aggt Bond Index', alpha=0.5)
        self.weight_ax.plot(self.df_info['weights4'],  label='S&P 500 IT Index', alpha=0.5)
        self.weight_ax.plot(self.df_info['weights5'],  label='Municipal Bond', alpha=0.5)
        self.weight_ax.plot(self.df_info['weights6'],  label='MSCI Pacific exjp ETF', alpha=0.5)
        self.weight_ax.plot(self.df_info['weights7'],  label='MSCI Emerging Markets', alpha=0.5)
        self.weight_ax.set_title("Weight")
        self.weight_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=4)
        plt.ylim(-0.05, 0.6)

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

class DataGenerator(object):
    def __init__(self, history, abbreviation, steps, window_length=50, start_idx=0, start_date=None):
        assert history.shape[0] == len(abbreviation), 'Number of stock is consistent'
        import copy

        self.steps = steps + 1
        self.window_length = window_length
        self.start_idx = start_idx
        self.start_date = start_date
        self._data = history.copy()  
        self.asset_names = copy.copy(abbreviation)


    def _step(self):
        self.step += 1
        obs = self.data[:, self.step:self.step + self.window_length, :].copy()
        obs_ = self.data[:, self.step-1:self.step + self.window_length, :].copy()

        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy() 
        done = self.step >= self.steps

        return obs, done, ground_truth_obs, obs_

    def reset(self):
        self.step = 0
        if self.start_date is None:
            self.idx = np.random.randint(  
                low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            self.idx = date_to_index(self.start_date) - self.start_idx
            
        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :4]
        self.data = data
        return self.data[:, self.step:self.step + self.window_length, :].copy(), \
               self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()


class PortfolioSim(object):
    def __init__(self, steps, asset_names=list(), trading_cost=0.0001, time_cost=0.0):
        self.asset_names = asset_names
        self.trading_cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps


    def _step(self, w1, y1, st):
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0
        p0 = self.p0
        b0 = self.b0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + EPS)            
        s1 = self.trading_cost * (np.abs(dw1 - w1)).sum()      

        total_wealth = 1 + ADD_INVEST[0:st-1].sum()
        assert s1 < total_wealth, 'Cost is larger than current holding'

        p1 = (p0 + ADD_INVEST[st-1]) * (1 - s1) * np.dot(y1, w1) 
        p1 = p1 * (1 - self.time_cost)     
        rho1 = (p1 / p0) - 1               
        r1 = np.log((p1 + EPS) / (p0 + EPS))  
        
        df_info = pd.read_csv(CSV_DIR)
        df_info = market_value(df_info, self.trading_cost)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        
        rr = np.rate(nper=self.steps, pmt=-ADD_INVEST[0], pv=-1, fv=GOAL)
        b1 = (b0 + ADD_INVEST[st-1]) * (1 - s1) * np.dot(y1, np.repeat(w1.mean(),7))
        P = (1-RHO)*b1 + RHO*GAMMA_r*p0    
    
        # reward function    
        reward =  -(ALPHA**(min(p1-GOAL,0))) + BETA*(p1-b1) + GAMMA*sharpe_ratio + DELTA*(w1[0]+ w1[2]+w1[4])*(st / self.steps) - THETA*mdd - MU*ADD_INVEST[st-1] 
        # remember for next step
        self.w0 = w1
        self.p0 = p1
        self.b0 = b1

        done = bool(p1 < 0.9*total_wealth)

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": np.sum((1/(len(self.asset_names)+1)) * (y1-1)), 
            "rate_of_return": rho1,
            "weights1": w1[0],
            "weights2": w1[1],
            "weights3": w1[2],
            "weights4": w1[3],
            "weights5": w1[4],
            "weights6": w1[5],
            "weights7": w1[6],
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": s1,
            "benchmark": b1,
            "add_invest": ADD_INVEST[st-1],
            "total_add": ADD_INVEST[0:st-1].sum(),
            "sp":self.steps,
            'y1':y1,
            "target":P,
            "rr":rr,
            "st":st,
            "total_wealth":total_wealth
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.asset_names))
        self.p0 = 1.0
        self.b0 = 1.0