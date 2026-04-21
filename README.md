# High-Frequency Limit Order Book (LOB) Simulator & Market Making Engine

A comprehensive, event-driven market microstructure simulator, predictive machine learning engine, and vectorized backtester. This system simulates a live financial exchange, engineers quantitative signals, trains a non-linear Alpha model (XGBoost) to predict short-term price movements, and executes a signal-assisted Market Making strategy under realistic market frictions.

## 🚀 System Architecture
* **Core Matching Engine:** Utilizes $O(1)$ time-complexity `collections.deque` structures to process high-frequency queues of Bids and Asks.
* **Data Feed:** A stochastic market maker that seeds the initial book and simulates rapid-fire chaotic market events (liquidity provision and taking).
* **Crossed-Book Prevention:** Advanced routing logic that instantly executes aggressive limit orders that cross the spread, perfectly mirroring live exchange matching engines.

## 🧮 Microstructure Feature Engineering
The engine extracts real-time mathematical signals from the shape of the LOB to predict order flow toxicity:
* **Order Book Imbalance (OBI):** The ratio of resting buy volume to sell volume at the top of the book.
* **Volume-Weighted Micro-Price:** Adjusts the theoretical Mid-Price by anchoring it to the heaviest liquidity mass.
* **Bid-Ask Spread Dynamics:** Tracks liquidity gaps to forecast incoming volatility.

## 🤖 Alpha Model (XGBoost)
* **Data Pipeline:** Captures millisecond-level snapshots of the engineered signals into a Pandas time-series DataFrame.
* **Strict Chronological Validation:** Completely eliminates lookahead bias by enforcing a strict time-based train/test split (`shuffle=False`), preventing data leakage.
* **Model:** Upgraded from baseline Logistic Regression to an `XGBClassifier` to capture complex, non-linear interactions within the microstructure data, achieving ~67% out-of-sample accuracy.

## 📉 Robustness Analysis & Adverse Selection
A naive backtest executing aggressively at the Mid-Price yielded an unrealistic Sharpe Ratio. To stress-test the Alpha model, strict execution frictions were introduced (crossing the spread, taker fees). As expected in real-world HFT, crossing the spread destroyed the statistical edge.

To survive execution realities, the strategy was pivoted to **Passive Market Making** with simulated **Adverse Selection**:
* **Signal-Assisted Quoting:** The algorithm passively quotes the Bid or Ask, attempting to capture the spread.
* **Adverse Selection Penalty:** The backtester assumes a low fill rate (~35%) on winning quotes (missing the wave) and a 100% fill rate on losing quotes (catching a falling knife).
* **Defensive Alpha:** The XGBoost model probabilities are used as a confidence threshold (>60%) to dictate which side of the book to quote, protecting the market maker from toxic flow.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Data & Backtesting:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib

## ⚡ Execution
To boot up the exchange, simulate 5,000 high-frequency events, train the XGBoost model, run the adverse-selection backtest, and generate the PnL equity curve chart:
```bash
python main.py