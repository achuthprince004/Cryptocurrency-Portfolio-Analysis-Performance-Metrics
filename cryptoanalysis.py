import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1️⃣ Fetch historical crypto data (~6 years)
# --------------------------
tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "USDT-USD"]

price_data = yf.download(
    tickers,
    start="2019-01-01",
    end="2025-08-01"
)["Close"]  

price_data.dropna(inplace=True)

# --------------------------
# 2️⃣ Calculate daily returns
# --------------------------
returns = price_data.pct_change().dropna()

# --------------------------
# 3️⃣ Define portfolios
# --------------------------
portfolios = {
    "100% BTC": [1, 0, 0, 0, 0],
    "50% BTC + 50% ETH": [0.5, 0.5, 0, 0, 0],
    "33% BTC + 33% ETH + 33% SOL": [1/3, 1/3, 1/3, 0, 0],
}

# --------------------------
# 4️⃣ Portfolio simulation function
# --------------------------
def simulate_portfolio(weights):
    weights = pd.Series(weights, index=returns.columns)
    port_return = (returns * weights).sum(axis=1)
    cumulative = (1 + port_return).cumprod()
    return cumulative

# --------------------------
# 5️⃣ Plot portfolio performance
# --------------------------
plt.figure(figsize=(12, 6))
for name, weights in portfolios.items():
    cumulative = simulate_portfolio(weights)
    plt.plot(cumulative, label=name)

plt.title("Portfolio Performance Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------
# 6️⃣ Metrics functions
# --------------------------
def sharpe_ratio(return_series, risk_free_rate=0.005):
    excess_returns = return_series.mean() - risk_free_rate / 365
    return np.sqrt(365) * excess_returns / return_series.std()

def max_drawdown(series):
    peak = series.cummax()
    drawdown = (series - peak) / peak
    return drawdown.min()

# --------------------------
# 7️⃣ Calculate metrics for each portfolio
# --------------------------
sharpe_values = {}
drawdown_values = {}

for name, weights in portfolios.items():
    weights = pd.Series(weights, index=returns.columns)
    port_return = (returns * weights).sum(axis=1)
    cumulative = simulate_portfolio(weights)

    sharpe_values[name] = sharpe_ratio(port_return)
    drawdown_values[name] = max_drawdown(cumulative)

# --------------------------
# 8️⃣ Sharpe Ratio Bar Chart
# --------------------------
plt.figure(figsize=(8, 5))
plt.bar(sharpe_values.keys(), sharpe_values.values(), color='skyblue')
plt.title("Sharpe Ratio by Portfolio")
plt.ylabel("Sharpe Ratio")
plt.grid(axis='y')
plt.show()

# --------------------------
# 9️⃣ Max Drawdown Bar Chart
# --------------------------
plt.figure(figsize=(8, 5))
plt.bar(drawdown_values.keys(), drawdown_values.values(), color='salmon')
plt.title("Max Drawdown by Portfolio")
plt.ylabel("Max Drawdown")
plt.grid(axis='y')
plt.show()
