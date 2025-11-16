from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env_hedging import HedgingEnv
from data_loader import get_nse_prices, make_price_paths_from_series

def make_env_with_real_data():
    prices = get_nse_prices("RELIANCE.NS", start="2015-01-01", end="2024-12-31")
    paths = make_price_paths_from_series(prices, window_size=50)
    def _init():
        return HedgingEnv(price_paths=paths, K=100)  

    return _init

if __name__ == "__main__":
    env = DummyVecEnv([make_env_with_real_data()])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=2048,
        n_steps=2048,
    )

    model.learn(total_timesteps=500_000)
    model.save("ppo_hedging_realdata")
