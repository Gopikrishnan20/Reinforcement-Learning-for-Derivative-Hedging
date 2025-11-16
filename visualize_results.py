import matplotlib.pyplot as plt
import numpy as np

def plot_error_histogram(rl_errors, delta_errors):
    plt.figure(figsize=(8,5))
    plt.hist(rl_errors, bins=40, alpha=0.6, label="RL Hedging")
    plt.hist(delta_errors, bins=40, alpha=0.6, label="Delta Hedging")
    plt.title("Hedging Error Distribution")
    plt.xlabel("Hedging Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cost_vs_error(rl_errors, rl_costs, delta_errors, delta_costs):
    plt.figure(figsize=(8,5))
    plt.scatter(rl_costs, rl_errors, alpha=0.6, label="RL", color="blue")
    plt.scatter(delta_costs, delta_errors, alpha=0.6, label="Delta", color="red")
    plt.title("Cost vs Hedging Error")
    plt.xlabel("Transaction Cost")
    plt.ylabel("Hedging Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_price_and_positions(price_path, rl_positions, delta_positions, rl_portfolio, delta_portfolio):
    fig, axs = plt.subplots(3, 1, figsize=(10,10))

    axs[0].plot(price_path)
    axs[0].set_title("Underlying Price Path")

    axs[1].plot(rl_positions, label="RL Positions")
    axs[1].plot(delta_positions, label="Delta Positions")
    axs[1].set_title("Hedge Positions Over Time")
    axs[1].legend()

    axs[2].plot(rl_portfolio, label="RL Portfolio")
    axs[2].plot(delta_portfolio, label="Delta Portfolio")
    axs[2].set_title("Portfolio Value Over Time")
    axs[2].legend()

    plt.tight_layout()
    plt.show()
