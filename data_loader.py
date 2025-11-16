import yfinance as yf
import numpy as np

def get_nse_prices(ticker="RELIANCE.NS",
                   start="2015-01-01",
                   end="2024-12-31"):
    data = yf.download(ticker, start=start, end=end)
    prices = data["Close"].dropna().values.astype(float)
    return prices


def make_price_paths_from_series(price_series, window_size=50):
    paths = []
    for i in range(len(price_series) - window_size):
        path = price_series[i : i + window_size + 1]
        paths.append(path)
    return np.array(paths)  
