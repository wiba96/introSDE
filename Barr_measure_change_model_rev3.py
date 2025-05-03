import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

# === Streamlit App Interface ===
st.set_page_config(page_title="Measure Change Financial Simulator", layout="wide")
st.title("📈 Measure Change in Financial Modeling")

st.markdown("""
## 📘 Background
This app simulates the future paths of a stock price and estimates the value of a financial derivative called a **European call option**.

To do this, we use a technique from quantitative finance called **Geometric Brownian Motion (GBM)**, which is a mathematical model for simulating how asset prices evolve over time.

We simulate under two different **probability measures**:

- A **real-world measure (P)** which reflects actual investor behavior and market dynamics.
- A **risk-neutral measure (Q)** used for pricing financial derivatives, where the expected return of an asset is the risk-free interest rate.

In both cases, the price process has a drift (expected growth rate) and volatility (randomness).

- The **drift** is the average rate at which an asset price grows.
- Under the **real-world measure**, the drift is based on historical average return (μ).
- Under the **risk-neutral measure**, the drift is changed to the risk-free rate (r), allowing us to calculate fair prices for options.

This shift in measure is justified by **Girsanov’s Theorem**, and our option price calculation follows the logic of the **Feynman-Kac formula**, which says that we can use expectations under the risk-neutral measure instead of solving complex differential equations.
""")

st.markdown("""
## 🔀 Simulation Types Explained
- **Real-World (μ):** Uses the average historical return of the asset as the expected drift. This represents how investors believe the stock will perform based on past performance. It's useful for forecasting and assessing market behavior.

- **Risk-Neutral (r):** Uses the risk-free rate as the expected return. This is not about predicting market behavior but about **pricing options** fairly based on the assumption that all investors are indifferent to risk. It ensures no arbitrage opportunities exist.

Switching between these gives insights into how assumptions about the future affect price expectations and risk estimates.

---
### 🎯 Option Type and Strike Price Explained
- **Option Type:**
    - **Call Option:** Gives the buyer the right (not obligation) to **buy** the asset at a set price (strike price) at a future date. It's profitable when the asset price goes **above** the strike price.
    - **Put Option:** Gives the buyer the right to **sell** the asset at the strike price. It's profitable when the asset price goes **below** the strike price.

  The choice of option type changes how the payoff is calculated in simulations. Calls benefit from rising markets, puts benefit from falling markets.

- **Strike Price:**
    The price at which the option can be exercised. It's the "threshold" for profit:
    - A **lower** strike price for a call option makes it easier to profit (but usually costs more).
    - A **higher** strike price for a put option increases potential profit if prices fall.

  In the simulation, changing the strike price directly influences the likelihood and size of payoff values in your option pricing calculation.
""")

# === Sidebar Controls (Updated for Revision 1 Enhancements) ===
symbol = st.sidebar.text_input("Stock Ticker Symbol", value="SPY")
period = st.sidebar.selectbox("Historical Period", ["1y", "3y", "5y", "10y"], index=2)
n_recent_days = st.sidebar.slider("Days of Historical Data to Display", min_value=30, max_value=252, value=100)
sim_type = st.sidebar.radio("Simulation Type", ["Real-World (μ)", "Risk-Neutral (r)"])
n_paths = st.sidebar.slider("Number of Simulation Paths", 100, 20000, 10000, step=100)
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])

# === Load Market Data ===
ticker = yf.Ticker(symbol)
data = ticker.history(period=period, interval="1d")

if data.empty:
    st.error("Failed to fetch data. Please check the symbol or internet connection.")
    st.stop()

close_prices = data["Close"].dropna()
recent_prices = close_prices[-n_recent_days:]
S0 = close_prices.iloc[-1]
strike_price_input = st.sidebar.number_input("Strike Price", min_value=1.0, value=float(S0), step=1.0, format="%0.2f")

# === Parameters ===
log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
mu = log_returns.mean() * 252
sigma = log_returns.std() * np.sqrt(252)
T = 1.0
r = 0.03
n_steps = 252
dt = T / n_steps

# === GBM Simulation ===
def simulate_gbm(S0, drift, sigma, T, n_steps, n_paths):
    S = np.zeros((n_steps + 1, n_paths))
    S[0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        S[t] = S[t - 1] * np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S

sim_drift = mu if sim_type == "Real-World (μ)" else r
S_sim = simulate_gbm(S0, sim_drift, sigma, T, n_steps, n_paths)

# === Option Pricing ===
K = strike_price_input
if option_type == "Call":
    payoff = np.maximum(S_sim[-1] - K, 0)
else:
    payoff = np.maximum(K - S_sim[-1], 0)
option_price = np.exp(-r * T) * np.mean(payoff)

st.markdown(
    f"""
### 💰 Simulated European {option_type} Option Price: **${option_price:.2f}**
This is the estimated fair price of a European {option_type.lower()} option with a strike price equal to the current stock price (**${S0:.2f}**).

A **European {option_type.lower()} option** gives the holder the right (but not the obligation) to {"buy" if option_type == "Call" else "sell"} the asset at the strike price on a specific date in the future (in this case, 1 year from now).

If the simulated price at expiration is {"higher than" if option_type == "Call" else "lower than"} the strike price, the option has value (you profit by {"buying low and selling high" if option_type == "Call" else "selling high before buying lower"}). If the price is {"lower" if option_type == "Call" else "higher"}, the option expires worthless.

The value shown is the average payoff across all simulated future price paths (under the chosen measure), discounted to present value using the risk-free rate. It reflects the **expected future profit** from this option, accounting for both gains and losses.

> **Example:**
If the current {symbol} price is \${S0:.2f} and the {option_type.lower()} option is priced at \${option_price:.2f}, you would pay \${option_price:.2f} today for the **chance** to {"buy" if option_type == "Call" else "sell"} {symbol} at \${K:.2f} in one year — expecting the price to be significantly {"higher" if option_type == "Call" else "lower"} on average under the simulated paths.
"""
)

# === Plot Simulated Paths ===
st.subheader(f"Simulated {sim_type} Paths + Historical Data for {symbol}")
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(10):
    ax.plot(np.arange(n_recent_days, n_recent_days + n_steps + 1), S_sim[:, i], lw=1, alpha=0.6)
ax.plot(np.arange(n_recent_days), recent_prices.values, color='black', label='Historical Prices')
ax.set_xlabel("Trading Days")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{sim_type} Simulation with Historical Prices")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# === Confidence Interval Plot ===
st.subheader("📊 Confidence Interval of Simulated Prices")
percentiles = np.percentile(S_sim, [5, 50, 95], axis=1)
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(percentiles[1], label="Median Path", color="blue")
ax2.fill_between(range(len(percentiles[0])), percentiles[0], percentiles[2], color="blue", alpha=0.2, label="5-95% Range")
ax2.set_title("Simulated Price Paths: Median and Confidence Interval")
ax2.set_xlabel("Days")
ax2.set_ylabel("Price")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# === Risk Metrics ===
def compute_var_cvar(returns, confidence=0.95):
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar

log_returns = np.log(S_sim[-1] / S0)
var_95, cvar_95 = compute_var_cvar(log_returns, 0.95)
var_95_pretty = var_95 * 100
cvar_95_pretty = cvar_95 * 100

st.subheader("📉 Risk Metrics")
col1, col2 = st.columns(2)
col1.metric("95% Value at Risk (VaR)", f"{var_95_pretty:.2f}%")
col2.metric("95% Conditional VaR (CVaR)", f"{cvar_95_pretty:.2f}%")

st.markdown("""
---
### 📚 Risk Metric Definitions
- **Value at Risk (VaR):** The maximum expected loss over a one-year period with 95% confidence. For example, if VaR = -18.00%, there’s a 95% chance your portfolio won't lose more than 18%.

- **Conditional Value at Risk (CVaR):** The average loss *given* that the loss exceeds the VaR. It tells us how bad the worst 5% of cases could be, offering insight into tail risk.
""")

st.markdown("""
---
### 🧭 What Does This Information Mean to You?

This simulation gives you a data-driven way to explore how uncertainty, risk, and pricing work together in financial markets. Here are some things you might take away and next steps to consider:

- **If you're an investor:** Use the real-world simulation to explore how your portfolio might behave under current market trends. If the projected paths show consistent growth, it could affirm long-term positions. Use the risk metrics to understand potential downside exposure.

- **If you're considering options:** The estimated call option price helps you understand what the market might consider a fair price to bet on future growth. Compare this simulated value with real market quotes to find underpriced or overpriced opportunities.

- **If you're a student or researcher:** This tool demonstrates how measure changes (like Girsanov’s theorem) and risk-neutral pricing (via Feynman-Kac) apply to real market data. Experiment with different tickers, paths, and simulation types to deepen your understanding.

- **If you're building risk models:** Use the VaR and CVaR outputs to gauge the likelihood and severity of losses over a simulated year. These measures are critical in risk management and regulatory compliance.

**Next steps:**
- Try different tickers to compare behaviors across industries.
- Adjust the time frame to explore short-term vs. long-term dynamics.
- Explore real option pricing tools (e.g., Black-Scholes or binomial models) and compare their results with these simulations.
- Consider using this as a foundation to build trading signals, hedging strategies, or valuation dashboards.

Ultimately, this tool is meant to turn abstract finance concepts into something you can interact with, analyze, and learn from.
""")
