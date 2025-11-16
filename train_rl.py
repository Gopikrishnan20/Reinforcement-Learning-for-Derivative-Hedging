from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env_hedging import HedgingEnv

def make_env():
    return HedgingEnv()

env = DummyVecEnv([make_env])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=300000)

model.save("ppo_hedging")

