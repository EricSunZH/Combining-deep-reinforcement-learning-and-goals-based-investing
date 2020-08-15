import os
os.getcwd()
import numpy as np
from train_env import PortfolioEnv

env = PortfolioEnv(steps=119, trading_cost=0.0001, time_cost=0.00,window_length=5,start_idx=0,sample_start_date='2000-01-09')

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import warnings
warnings.filterwarnings('ignore')

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

log_dir = "./"
os.makedirs(log_dir, exist_ok=True)

n_actions =  env.action_space.shape[-1]  
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

env = Monitor(env, log_dir)
model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)  
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)  

time_steps = 10000
model.learn(total_timesteps=time_steps, callback=callback) 
model.save("ddpg_portfolio")
os.removedirs('best_model')
model = DDPG.load('best_model')

obs = env.reset()
action, _states = model.predict(obs)
obs, rewards, dones, info = env.step(action)
env.render()
