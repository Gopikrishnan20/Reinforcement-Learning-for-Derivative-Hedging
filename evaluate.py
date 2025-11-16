import numpy as np
from stable_baselines3 import PPO

from env_hedging import HedgingEnv
from baseline_delta import delta_hedging
from data_loader import get_nse_prices, make_price_paths_from_series
from visualize_results import plot_error_histogram, plot_cost_vs_error


def eval_rl_on_real_data(model, price_paths, n_episodes=300):
    errors = []
    costs = []

    env = HedgingEnv(price_paths=price_paths)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated

        payoff = max(env.S - env.K, 0.0)
        hedging_error = env.portfolio_value - payoff
        errors.append(hedging_error)
        costs.append(env.total_cost)

    return np.array(errors), np.array(costs)


def eval_delta_on_real_data(price_paths):
    errors = []
    costs = []
    for path in price_paths:
        path = path.astype(float)
        S0 = float(path[0])
        he, c = delta_hedging(path, K=S0)  
        errors.append(he)
        costs.append(c)
    return np.array(errors), np.array(costs)


if __name__ == "__main__":
    prices = get_nse_prices("RELIANCE.NS") 
    price_paths = make_price_paths_from_series(prices, window_size=50)
    model = PPO.load("ppo_hedging_realdata_v2.zip")  
    rl_errors, rl_costs = eval_rl_on_real_data(model, price_paths, n_episodes=300)
    print("RL (real-data):")
    print("  Mean error:", rl_errors.mean())
    print("  Std error :", rl_errors.std())
    print("  Mean cost :", rl_costs.mean())

    delta_errors, delta_costs = eval_delta_on_real_data(price_paths)
    print("Delta:")
    print("  Mean error:", delta_errors.mean())
    print("  Std error :", delta_errors.std())
    print("  Mean cost :", delta_costs.mean())

    plot_error_histogram(rl_errors, delta_errors)
    plot_cost_vs_error(rl_errors, rl_costs, delta_errors, delta_costs)
