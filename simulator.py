import numpy as np

def simulate_gbm(S0, mu, sigma, T, n_steps):
    dt = T / n_steps
    prices = [S0]
    for _ in range(n_steps):
        z = np.random.randn()
        S_next = prices[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
        prices.append(S_next)
    return np.array(prices)

