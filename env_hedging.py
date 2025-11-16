import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm

from simulator import simulate_gbm

def bs_call_price(S, K, r, sigma, T):
    if T <= 0:
        return max(S - K, 0.0)
    if S <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


class HedgingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        S0=100.0,
        K=None,
        mu=0.0,
        sigma=0.2,
        T=1.0,
        n_steps=50,
        cost_rate=0.001,
        max_position=1.0,
        price_paths=None,      
        error_weight=20.0,     
        cost_weight=0.1,       
        tail_threshold=0.05,   
        tail_weight=50.0,      
        risk_free_rate=0.0,
    ):
        super().__init__()
        self.S0_init = float(S0)
        self.K_init = None if K is None else float(K)
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.n_steps_default = n_steps
        self.cost_rate = cost_rate
        self.max_position = max_position
        self.r = risk_free_rate

        self.error_weight = error_weight
        self.cost_weight = cost_weight
        self.tail_threshold = tail_threshold  
        self.tail_weight = tail_weight
        self.price_paths = price_paths
        low = np.array(
            [0.0, 0.0, -max_position, -10.0, 0.0, -1.0, -1.0, -1.0],
            dtype=np.float32,
        )
        high = np.array(
            [10.0, 1.0, max_position, 10.0, 10.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.max_delta = 0.1
        self.action_space = spaces.Box(
            low=-self.max_delta, high=self.max_delta, shape=(1,), dtype=np.float32
        )
        self.path = None
        self.n_steps = None
        self.S0 = None
        self.K = None
        self.t = 0
        self.S = 0.0
        self.position = 0.0
        self.total_cost = 0.0
        self.portfolio_value = 0.0
        self.premium = 0.0
        self.price_history = []
        self.position_history = []
        self.portfolio_history = []
        self.cost_history = []
        self.return_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.price_paths is not None:
            idx = self.np_random.integers(0, len(self.price_paths))
            self.path = self.price_paths[idx].astype(float)
            self.n_steps = len(self.path) - 1
            self.S0 = float(self.path[0])
        else:
            self.S0 = self.S0_init
            self.path = simulate_gbm(self.S0, self.mu, self.sigma,
                                     self.T, self.n_steps_default)
            self.n_steps = len(self.path) - 1

        self.K = self.S0 if self.K_init is None else self.K_init

        self.t = 0
        self.S = float(self.path[0])
        self.position = 0.0
        self.total_cost = 0.0

        self.premium = bs_call_price(self.S0, self.K, self.r, self.sigma, self.T)
        self.portfolio_value = self.premium

        self.price_history = [self.S]
        self.position_history = [self.position]
        self.portfolio_history = [self.portfolio_value]
        self.cost_history = [self.total_cost]
        self.return_history = []

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        S_norm = self.S / self.S0 if self.S0 != 0 else 0.0
        pv_norm = self.portfolio_value / self.S0 if self.S0 != 0 else 0.0
        cost_norm = self.total_cost / self.S0 if self.S0 != 0 else 0.0
        t_norm = self.t / self.n_steps if self.n_steps > 0 else 0.0
        r1 = self.return_history[-1] if len(self.return_history) >= 1 else 0.0
        r2 = self.return_history[-2] if len(self.return_history) >= 2 else 0.0
        r3 = self.return_history[-3] if len(self.return_history) >= 3 else 0.0

        return np.array(
            [S_norm, t_norm, self.position, pv_norm, cost_norm, r1, r2, r3],
            dtype=np.float32,
        )

    def step(self, action):
        delta_pos = float(np.clip(action[0], -self.max_delta, self.max_delta))
        new_pos = float(
            np.clip(self.position + delta_pos, -self.max_position, self.max_position)
        )
        trans_cost = self.cost_rate * abs(new_pos - self.position) * self.S
        self.total_cost += trans_cost

        S_next = float(self.path[self.t + 1])

        if self.S > 0:
            ret = np.log(S_next / self.S)
        else:
            ret = 0.0
        self.return_history.append(ret)

        pnl_underlying = new_pos * (S_next - self.S)
        self.portfolio_value += pnl_underlying - trans_cost
        self.position = new_pos
        self.S = S_next
        self.t += 1

        self.price_history.append(self.S)
        self.position_history.append(self.position)
        self.portfolio_history.append(self.portfolio_value)
        self.cost_history.append(self.total_cost)

        terminated = False
        reward = - (trans_cost / max(self.S0, 1.0))  

        if self.t == self.n_steps:
            payoff = max(self.S - self.K, 0.0)  
            hedging_error = self.portfolio_value - payoff
            scaled_error = hedging_error / max(self.S0, 1.0)
            scaled_cost = self.total_cost / max(self.S0, 1.0)
            base_penalty = self.error_weight * (scaled_error ** 2)
            tail_penalty = 0.0
            if hedging_error < -self.tail_threshold * self.S0:
                tail_penalty = self.tail_weight * abs(scaled_error)

            final_penalty = base_penalty + tail_penalty + self.cost_weight * scaled_cost
            reward = -final_penalty
            terminated = True

        obs = self._get_obs()
        info = {
            "S": self.S,
            "K": self.K,
            "portfolio_value": self.portfolio_value,
            "total_cost": self.total_cost,
        }
        return obs, reward, terminated, False, info
