# Deep Reinforcement Learning for Derivative Hedging
A implementation of Deep Hedging using Reinforcement Learning (PPO), benchmarked against classical Delta Hedging, and evaluated on both GBM-simulated markets and real stock market data (NSE India).

This project demonstrates how RL can learn cost-efficient hedging strategies under transaction costs and realistic market dynamics.

**Highlights**

Custom OpenAI Gym environment for derivative hedging
Trains a PPO agent to hedge an option using underlying price movements
Supports GBM simulation and real historical stock data
Built-in Delta Hedging baseline for fair benchmarking
Comprehensive evaluation: hedging error, cost–risk tradeoff, and visual analytics
Clean, modular codebase — easy to extend or integrate into research pipelines

**Concept Overview**

In theory, Delta hedging provides the optimal hedge in a frictionless Black–Scholes world.
In reality:
Markets have transaction costs
Prices evolve discretely
Volatility varies
Continuous rebalancing is impossible
Deep Hedging allows an RL agent to learn hedging strategies directly from data — without assuming a perfect model — enabling a more flexible cost–risk tradeoff.
This project replicates this idea using PPO.

**Project Structure**
rl-derivative-hedging/
 env_hedging.py              # Gym environment (real data + GBM)
 simulator.py                # GBM price generator
 baseline_delta.py           # Classical Delta Hedging implementation
 data_loader.py              # NSE data loader & path generator
 train_gbm.py                # Train PPO agent (GBM)
 train_realdata.py           # Train PPO agent (real data)
 evaluate_gbm.py             # Evaluation on GBM
 evaluate_realdata.py        # Evaluation on real data
 visualize_results.py        # Plotting utilities
 README.md                   # Project documentation

** Methodology**
**1. Market Data**
GBM simulation using drift μ and volatility σ
NSE historical pricing data (e.g., RELIANCE.NS) retrieved via yfinance
Time-windowed price paths (50-day rolling windows)
**2. Hedging Environment**
Each episode simulates the full lifecycle of an option:
Agent adjusts its hedge position
Portfolio updated mark-to-market
Transaction costs applied
Payoff realized at maturity
Final hedging error computed
**3. Reward Design**
Penalizes:
Squared hedging error
Transaction costs
Tail-risk (large negative errors)
**4. RL Agent**
PPO (Stable Baselines 3)
Training for up to 1M timesteps
Continuous action space (Δ position)
Normalized observations and reward shaping
**5. Baseline**
Delta hedging is implemented using Black–Scholes Delta:
Δ = N(d1)

**Results Summary**
**Hedging Error Distribution**
Delta hedging shows tight clustering around zero
PPO agent shows larger variance but lower cost
**Cost vs Hedging Error**
RL strategies achieve significantly lower cost
Delta achieves lower risk (error)
Demonstrates the classic cost–risk tradeoff

**Technologies**

Python

Stable Baselines 3 (PPO)

Gymnasium

NumPy, Pandas

Matplotlib

yfinance
