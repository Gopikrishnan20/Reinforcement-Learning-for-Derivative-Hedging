import numpy as np
from scipy.stats import norm

def bs_delta(S, K, r, sigma, tau):
    if tau <= 0:
        return 0
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*tau) / (sigma*np.sqrt(tau))
    return norm.cdf(d1)

def delta_hedging(S_path, K=100, sigma=0.2, r=0, cost_rate=0.001, T=1):
    n_steps = len(S_path)-1
    dt = T / n_steps

    portfolio = 10.0
    position = 0.0
    total_cost = 0.0

    for t in range(n_steps):
        S = S_path[t]
        tau = T - t*dt
        delta_star = bs_delta(S, K, r, sigma, tau)
        delta_change = delta_star - position
        trans_cost = cost_rate * abs(delta_change) * S
        total_cost += trans_cost

        position = delta_star

        S_next = S_path[t+1]
        pnl = position * (S_next - S)
        portfolio += pnl - trans_cost

    payoff = max(S_path[-1] - K, 0)
    hedging_error = portfolio - payoff

    return hedging_error, total_cost

