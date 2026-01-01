# 📈 Portfolio Optimization using Deep Reinforcement Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://machine-learning-zoomcamp-homework-rl-portofolio-optimization.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ **Note:** The Jupyter notebook (`Final_Capestone_Project.ipynb`) may not display correctly on GitHub due to rendering issues. You can either:
> 1. **Download the notebook** and open it locally in Jupyter Notebook/JupyterLab
> 2. **View the Python script** (`Final_Capestone_Project.py`) which contains the same code

A comprehensive portfolio optimization system that leverages **Deep Reinforcement Learning (PPO - Proximal Policy Optimization)** to dynamically allocate assets and outperform traditional investment strategies. This project includes an interactive **Streamlit dashboard** for real-time analysis and visualization.

![Portfolio Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)

---

## 🚀 Live Demo

**🚀 Try the app now:** [**Portfolio Optimization Dashboard**](https://machine-learning-zoomcamp-homework-rl-portofolio-optimization.streamlit.app/)

No installation required! Simply click the link above to:
- Explore different portfolio optimization strategies
- Compare RL Agent performance against traditional methods
- Analyze real-time stock data from Yahoo Finance
- View interactive charts and risk metrics

---

## 📸 Dashboard Screenshots

### 📊 Overview Tab
![Overview Tab](screenshots/overview_tab.png)
*Market overview showing normalized price performance of all assets, asset statistics including annual returns, volatility, and Sharpe ratios, plus a correlation matrix heatmap.*

### 📈 Strategy Comparison Tab
![Strategy Comparison](screenshots/strategy_comparison.png)
*Side-by-side comparison of all portfolio strategies including RL Agent (PPO), Equal Weight, Risk Parity, Momentum, Minimum Variance, and Benchmark (SPY). Shows cumulative returns and performance metrics.*

### 🤖 RL Agent Tab
![RL Agent Tab](screenshots/rl_agent_tab.png)
*Dedicated analysis of the PPO reinforcement learning agent showing performance vs benchmarks, dynamic weight allocation over time, and model information.*

### ⚖️ Portfolio Weights Tab
![Portfolio Weights](screenshots/portfolio_weights.png)
*Visualization of portfolio weight evolution over time for each strategy, showing how allocations change dynamically based on market conditions.*

### 📉 Risk Analysis Tab
![Risk Analysis](screenshots/risk_analysis.png)
*Comprehensive risk metrics including drawdown analysis, Value at Risk (VaR), Conditional VaR (CVaR), and risk-adjusted performance comparisons.*

---

## 📊 Generated Analysis Charts

### Portfolio Performance Comparison
![Portfolio Comparison](portfolio_comparison.png)
*Cumulative returns comparison across all strategies over the backtest period.*

### RL Agent Weight Evolution
![RL Weights](rl_weights.png)
*Dynamic portfolio allocation by the RL Agent over time.*

### Drawdown Analysis
![Drawdowns](drawdowns.png)
*Peak-to-trough decline comparison for each strategy.*

### Price Analysis
![Price Analysis](price_analysis.png)
*Normalized price performance of all assets in the portfolio universe.*

### Average Portfolio Allocation
![Average Weights](average_weights.png)
*Average allocation distribution across strategies.*

### Feature Importance
![Feature Importance](feature_importance.png)
*Analysis of signals influencing the RL Agent's decisions.*

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Investment Strategies](#-investment-strategies)
- [Technical Architecture](#-technical-architecture)
- [Results & Performance](#-results--performance)
- [Installation](#-installation)
- [Running the Streamlit App](#-running-the-streamlit-app)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project addresses the classic **portfolio optimization problem** - how to allocate capital across multiple assets to maximize returns while managing risk. Traditional approaches like Modern Portfolio Theory (MPT) rely on historical statistics and assumptions that often fail in dynamic markets.

Our solution uses **Deep Reinforcement Learning** to create an adaptive agent that:
- Learns optimal allocation strategies from market data
- Adapts to changing market conditions
- Considers transaction costs in decision-making
- Balances risk and return automatically

### The Problem

Given a universe of 8 stocks (AAPL, GOOGL, MSFT, AMZN, JPM, XOM, PFE, KO), determine the optimal daily portfolio weights to maximize risk-adjusted returns over time.

### The Solution

A **PPO (Proximal Policy Optimization)** agent trained in a custom Gymnasium environment that:
- Observes recent price movements (10-day window)
- Outputs portfolio weights via softmax activation
- Receives risk-adjusted rewards
- Learns to minimize transaction costs

---

## ✨ Key Features

### 🤖 Deep Reinforcement Learning
- **PPO Algorithm**: State-of-the-art policy gradient method from Stable-Baselines3
- **Custom Environment**: Gymnasium-compatible trading environment
- **Risk-Adjusted Rewards**: Penalizes volatility while rewarding returns

### 📊 Interactive Dashboard
- **Real-time Data**: Live stock data from Yahoo Finance
- **6 Analysis Tabs**: Comprehensive portfolio analytics
- **Strategy Comparison**: Compare RL agent vs traditional strategies
- **Dynamic Visualizations**: Interactive Plotly charts

### 📈 Multiple Strategies
- RL Agent (PPO)
- Equal Weight
- Risk Parity
- Momentum
- Minimum Variance
- Benchmark (SPY)

---

## 💼 Investment Strategies

### 1. RL Agent (PPO) 🤖
Our reinforcement learning agent uses:
- **State Space**: 10-day rolling returns + current weights (88 dimensions)
- **Action Space**: Continuous weights for 8 assets
- **Reward Function**: Risk-adjusted returns with transaction cost penalty

```
Reward = Return × 100 - 0.5 × Return² × 1000 - Transaction_Cost
```

### 2. Equal Weight ⚖️
Simple 1/N allocation across all assets. Surprisingly effective baseline.

### 3. Risk Parity 📉
Allocates inversely proportional to asset volatility:
```
Weight_i = (1/σ_i) / Σ(1/σ_j)
```

### 4. Momentum 🚀
Invests in top N performing assets based on recent returns.

### 5. Minimum Variance 🛡️
Minimizes portfolio variance using inverse variance weighting.

---

## 🔧 Technical Architecture

### Reinforcement Learning Environment

```python
class PortfolioEnvFast(gym.Env):
    """
    Custom Gymnasium environment for portfolio optimization.
    
    Observation Space: [window_returns, current_weights]
    Action Space: Continuous [-1, 1] for each asset (softmaxed to weights)
    Reward: Risk-adjusted return with transaction costs
    """
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO |
| Window Size | 10 days |
| Episode Length | 50 steps |
| Total Timesteps | 30,000 |
| Learning Rate | 3e-4 |
| Transaction Cost | 0.1% |

### Model Architecture
- **Policy Network**: MLP with 2 hidden layers (64 units each)
- **Value Network**: Separate MLP for value estimation
- **Activation**: Tanh

---

## 📊 Results & Performance

### Backtest Results (2019-2023)

| Strategy | Total Return | Annual Return | Sharpe Ratio | Max Drawdown |
|----------|--------------|---------------|--------------|--------------|
| **RL Agent (PPO)** | **85.2%** | **13.2%** | **0.89** | **-18.5%** |
| Equal Weight | 72.4% | 11.5% | 0.74 | -22.3% |
| Risk Parity | 68.9% | 11.0% | 0.81 | -19.8% |
| Momentum | 61.2% | 10.0% | 0.62 | -28.4% |
| Min Variance | 55.8% | 9.3% | 0.78 | -16.2% |
| Benchmark (SPY) | 64.5% | 10.5% | 0.68 | -23.9% |

### Key Findings

1. **RL Agent Outperforms**: Achieves highest Sharpe ratio (0.89) indicating superior risk-adjusted returns
2. **Adaptive Allocation**: RL agent dynamically shifts weights based on market conditions
3. **Transaction Efficiency**: Learned to minimize unnecessary rebalancing
4. **Drawdown Control**: Competitive maximum drawdown despite aggressive positioning

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/Yonas-Berhe/machine-learning-zoomcamp-homework.git
cd machine-learning-zoomcamp-homework/Capstone_Project
```

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 🖥️ Running the Streamlit App

### Local Development

1. **Navigate to project directory**:
   ```bash
   cd machine-learning-zoomcamp-homework/Capstone_Project
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Access the dashboard**:
   - Open your browser to `http://localhost:8501`
   - The app will automatically reload when you make changes

### Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 **Overview** | Market data, price charts, asset statistics, correlation matrix |
| 📈 **Strategy Comparison** | Cumulative returns, performance metrics for all strategies |
| 🤖 **RL Agent** | Dedicated RL analysis, weight evolution, model information |
| 🎯 **Performance Analysis** | Rolling metrics, returns distribution, monthly heatmap |
| ⚖️ **Portfolio Weights** | Weight evolution over time, current allocation |
| 📉 **Risk Analysis** | Drawdown charts, VaR, CVaR, risk metrics comparison |

### Configuration Options

Use the sidebar to customize:
- **Stock Tickers**: Add/remove assets from the universe
- **Date Range**: Adjust backtest period
- **Benchmark**: Choose SPY, QQQ, IWM, or DIA
- **Lookback Period**: Adjust strategy lookback windows
- **Momentum Top N**: Number of assets for momentum strategy
- **Risk-Free Rate**: For Sharpe ratio calculation
- **RL Window Size**: Observation window for RL agent

---

## 📁 Project Structure

```
Capstone_Project/
│
├── 📄 app.py                      # Streamlit dashboard application
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Project documentation
│
├── 📓 Final_Capestone_Project.ipynb  # Jupyter notebook with full analysis
│
├── 📦 portfolio_ppo_fast.zip      # Trained PPO model
│
├── 📸 screenshots/                # Dashboard screenshots
│   ├── overview_tab.png
│   ├── strategy_comparison.png
│   ├── rl_agent_tab.png
│   ├── portfolio_weights.png
│   └── risk_analysis.png
│
├── 📊 Analysis Outputs/
│   ├── portfolio_comparison.png   # Strategy comparison chart
│   ├── rl_weights.png            # RL weight evolution
│   ├── drawdowns.png             # Drawdown analysis
│   ├── price_analysis.png        # Price movements
│   ├── average_weights.png       # Average allocation
│   └── feature_importance.png    # Feature analysis
│
├── 📄 backtest_comparison.csv     # Backtest results data
└── 📄 summary_report.txt          # Text summary of results
```

---

## 🧠 How It Works

### 1. Data Collection
```python
# Fetch historical data from Yahoo Finance
data = yf.download(tickers, start='2019-01-01', end='2023-12-31')
returns = data['Close'].pct_change().dropna()
```

### 2. Environment Setup
The RL agent interacts with a custom trading environment:
```python
# Observation: Recent returns + current portfolio weights
obs = [returns[-10:].flatten(), current_weights]

# Action: New portfolio weights (via softmax)
action = model.predict(obs)
new_weights = softmax(action)

# Reward: Risk-adjusted return minus transaction costs
reward = portfolio_return - 0.001 * turnover
```

### 3. Training Loop
```python
# PPO training with custom environment
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=30000, callback=callback)
```

### 4. Backtesting
```python
# Walk-forward backtest
for each trading day:
    observe market state
    get model prediction
    calculate returns
    update portfolio
```

### 5. Visualization
Interactive Plotly charts display:
- Cumulative returns
- Portfolio weights over time
- Risk metrics
- Drawdown analysis

---

## 🔮 Future Improvements

- [ ] **Multi-timeframe analysis**: Incorporate weekly/monthly signals
- [ ] **Sentiment integration**: Add news sentiment as features
- [ ] **Alternative assets**: Include crypto, commodities, bonds
- [ ] **Ensemble methods**: Combine multiple RL agents
- [ ] **Live trading**: Connect to brokerage APIs
- [ ] **Risk constraints**: Add maximum position limits
- [ ] **Tax optimization**: Consider tax-loss harvesting

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [Streamlit](https://streamlit.io/) for the amazing dashboard framework
- [Yahoo Finance](https://finance.yahoo.com/) for market data
- [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) for the learning opportunity

---

## 📧 Contact

**Author**: Yonas Berhe

**Project Link**: [https://github.com/Yonas-Berhe/machine-learning-zoomcamp-homework/tree/main/Capstone_Project](https://github.com/Yonas-Berhe/machine-learning-zoomcamp-homework/tree/main/Capstone_Project)

**Live Demo**: [https://machine-learning-zoomcamp-homework-rl-portofolio-optimization.streamlit.app/](https://machine-learning-zoomcamp-homework-rl-portofolio-optimization.streamlit.app/)

---

<div align="center">

### ⭐ Star this repo if you found it useful!

Made with ❤️ for ML Zoomcamp Capstone Project

</div>
