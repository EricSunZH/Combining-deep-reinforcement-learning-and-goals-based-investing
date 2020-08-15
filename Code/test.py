# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:45:19 2020

@author: Eric Sun
"""
from settings import *
import os
os.getcwd()
from test_env import TestEnv
from stable_baselines.bench import Monitor
from stable_baselines import DDPG
import warnings
warnings.filterwarnings('ignore')

portfolio_success = 0
benchamrk_success = 0

# number of monte carlo simulations
N_mc = 100 # 1000 runs takes more than 1 hour. Change to 100 for a smaller sample

for i in range(N_mc):
    env = TestEnv(steps=119, mc_i=i,trading_cost=0.0001, time_cost=0.00,
                  window_length=5,start_idx=0,sample_start_date='2000-01-09')
    log_dir = "./"    
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    model = DDPG.load('best_model')
    obs = env.reset()
    print('\ncurrent round of simulation:', i+1)
    for _ in range(120):
        try:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
        except:
            pass

    if info['portfolio_value']>=GOAL:
        portfolio_success += 1
    if info['benchmark']>=GOAL:
        benchamrk_success += 1

print("Goal:", GOAL)
print("portfolio success rate:",portfolio_success/N_mc) 
print("benchamrk success rate:",benchamrk_success/N_mc) 

