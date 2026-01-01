!pip install yfinance pandas numpy matplotlib seaborn gymnasium stable-baselines3 torch scipy scikit-learn plotly

# ============================================
# GOOGLE COLAB SETUP
# ============================================

# Install required packages
!pip install -q yfinance gymnasium stable-baselines3[extra] torch

# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch. cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected - running on CPU")

# Set device for Stable-Baselines3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")

# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib. pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import Tuple, Dict, List, Optional

# Finance
import yfinance as yf

# ML/RL
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

# Stable Baselines 3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Sklearn
from sklearn.preprocessing import StandardScaler

# Settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)
torch.manual_seed(42)

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")

# Select a diverse set of assets (stocks across different sectors)
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JPM', 'XOM', 'PFE', 'KO', 'SPY']

print("Downloading historical data...")
print(f"Tickers: {tickers}")

# Download historical data
data = yf.download(tickers, start='2018-01-01', end='2023-12-31', progress=True)

print(f"\nData downloaded successfully!")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Total trading days: {len(data)}")

# %%
# Extract price data
close_prices = data['Close'].copy() # Changed 'Adj Close' to 'Close'
close_prices = close_prices.dropna()

# Calculate daily returns
returns = close_prices.pct_change().dropna()

# Display basic statistics
print("=" * 60)
print("ASSET STATISTICS")
print("=" * 60)
print(f"\nClose Prices Shape: {close_prices.shape}")
print(f"Returns Shape: {returns.shape}")

print("\n--- Daily Returns Summary ---")
print(returns.describe().round(4))

# %%
# Visualize price data
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Normalized prices
normalized_prices = close_prices / close_prices.iloc[0] * 100
normalized_prices.plot(ax=axes[0], linewidth=1.5)
axes[0].set_title('Normalized Asset Prices (Base = 100)', fontsize=14)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Normalized Price')
axes[0].legend(loc='upper left', ncol=3)
axes[0].grid(True, alpha=0.3)

# Cumulative returns
cumulative_returns = (1 + returns).cumprod()
cumulative_returns.plot(ax=axes[1], linewidth=1.5)
axes[1].set_title('Cumulative Returns', fontsize=14)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Cumulative Return')
axes[1].legend(loc='upper left', ncol=3)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('price_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Remove SPY from asset list for trading (keep for benchmark)
asset_names = [t for t in tickers if t != 'SPY']
n_assets = len(asset_names)

print(f"Trading Assets: {asset_names}")
print(f"Number of Assets: {n_assets}")

# Asset returns (excluding SPY)
asset_returns = returns[asset_names]. copy()

# Use returns as features (simplified version)
features = returns.copy()

# Normalize features
scaler = StandardScaler()
features_normalized = pd.DataFrame(
    scaler.fit_transform(features),
    index=features. index,
    columns=features. columns
)

print(f"\nFeatures Shape: {features_normalized.shape}")
print(f"Asset Returns Shape: {asset_returns. shape}")

train_ratio = 0.8
split_idx = int(len(features_normalized) * train_ratio)

# Training data
train_features = features_normalized. iloc[:split_idx]
train_returns = asset_returns.iloc[:split_idx]

# Testing data
test_features = features_normalized.iloc[split_idx:]
test_returns = asset_returns.iloc[split_idx:]

# SPY benchmark returns
spy_returns = returns['SPY'].copy()
test_spy_returns = spy_returns.iloc[split_idx:]

print("=" * 60)
print("DATA SPLIT")
print("=" * 60)
print(f"Training Period: {train_features.index[0]. date()} to {train_features.index[-1].date()}")
print(f"Training Samples: {len(train_features)}")
print(f"\nTesting Period: {test_features. index[0].date()} to {test_features.index[-1].date()}")
print(f"Testing Samples: {len(test_features)}")

# ============================================
# OPTIMIZED PORTFOLIO ENVIRONMENT (FASTER)
# ============================================

class PortfolioEnvFast(gym.Env):
    """
    Optimized Portfolio Environment for faster training.
    Key optimizations:
    - Shorter episodes with random starting points
    - Simplified observation space
    - Efficient numpy operations
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        window_size: int = 10,
        episode_length: int = 100,  # Fixed episode length for faster training
        transaction_cost:  float = 0.001,
        initial_capital: float = 100000,
        random_start: bool = True
    ):
        super().__init__()

        self.features = features_df. values. astype(np.float32)
        self.returns = returns_df.values.astype(np.float32)
        self.n_assets = returns_df.shape[1]
        self.n_features = features_df.shape[1]
        self.window_size = window_size
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self. initial_capital = initial_capital
        self.random_start = random_start
        self.dates = features_df.index
        self.max_steps = len(self.features) - 1

        # Action space:  portfolio weights
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Simplified observation:  current window flattened + current weights
        obs_dim = (self.window_size * self.n_features) + self.n_assets
        self.observation_space = spaces. Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed:  Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Random starting point for training diversity
        if self.random_start:
            max_start = self.max_steps - self.episode_length - self.window_size
            if max_start > self.window_size:
                self.start_step = np.random.randint(self.window_size, max_start)
            else:
                self. start_step = self.window_size
        else:
            self.start_step = self.window_size

        self.current_step = self.start_step
        self.episode_step = 0
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.portfolio_value = self.initial_capital

        # History tracking
        self.portfolio_values = [self.initial_capital]
        self.weights_history = [self.weights.copy()]
        self.returns_history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # Feature window
        start_idx = max(0, self.current_step - self.window_size)
        feature_window = self.features[start_idx:self.current_step]. flatten()

        # Pad if necessary
        expected_size = self.window_size * self.n_features
        if len(feature_window) < expected_size:
            feature_window = np.pad(feature_window, (expected_size - len(feature_window), 0))

        # Combine with current weights
        obs = np.concatenate([feature_window, self.weights]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray):
        # Normalize weights
        weights = np.maximum(action, 0)
        weight_sum = weights.sum()
        new_weights = weights / weight_sum if weight_sum > 0 else np.ones(self.n_assets) / self.n_assets
        new_weights = new_weights.astype(np.float32)

        # Transaction cost
        turnover = np.sum(np.abs(new_weights - self.weights))
        cost = turnover * self.transaction_cost

        # Portfolio return
        asset_returns = self.returns[self.current_step]
        portfolio_return = np.dot(new_weights, asset_returns) - cost

        # Update state
        self.portfolio_value *= (1 + portfolio_return)
        self.weights = new_weights. copy()

        # Record history
        self.portfolio_values.append(self.portfolio_value)
        self.weights_history.append(new_weights.copy())
        self.returns_history.append(portfolio_return)

        # Advance
        self.current_step += 1
        self.episode_step += 1

        # Check termination (fixed episode length OR end of data)
        terminated = (self.episode_step >= self.episode_length) or (self.current_step >= self. max_steps)
        truncated = False

        # Reward:  scaled return with small shaping for risk
        reward = portfolio_return * 100

        # Small penalty for high concentration (encourages diversification)
        concentration = np.sum(new_weights ** 2)  # Herfindahl index
        reward -= concentration * 0.1

        info = {'portfolio_value': self.portfolio_value, 'weights': new_weights}

        return self._get_observation(), reward, terminated, truncated, info

    def get_episode_metrics(self) -> dict:
        if len(self.returns_history) == 0:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}

        returns_array = np.array(self.returns_history)
        values_array = np.array(self. portfolio_values)

        total_return = (values_array[-1] / values_array[0]) - 1
        n_days = len(returns_array)

        volatility = np.std(returns_array) * np.sqrt(252) if n_days > 1 else 0.001
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        sharpe_ratio = (annual_return - 0.02) / max(volatility, 0.001)

        peak = np.maximum. accumulate(values_array)
        drawdown = (peak - values_array) / peak
        max_drawdown = drawdown.max()

        win_rate = np.mean(returns_array > 0)

        return {
            'total_return':  total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trading_days': n_days
        }

print("Testing the Portfolio Environment...")

# Create environment
env = PortfolioEnvFast(
    features_df=train_features,
    returns_df=train_returns,
    window_size=50,
    transaction_cost=0.001
)

print(f"Observation Space: {env.observation_space.shape}")
print(f"Action Space: {env.action_space.shape}")
print(f"Number of Assets: {env.n_assets}")

# Test reset
obs, info = env.reset()
print(f"\nInitial Observation Shape: {obs.shape}")
print(f"Initial Portfolio Value: ${env.portfolio_value:,.2f}")

# Test a few random steps
print("\nTesting random actions...")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {i+1}:  Reward = {reward:.4f}, Portfolio Value = ${info['portfolio_value']:,.2f}")

print("\nEnvironment test passed!")

# ============================================
# FIXED TRAINING CALLBACK
# ============================================

class TrainingCallback(BaseCallback):
    """Custom callback to track training progress."""

    def __init__(self, eval_env_fn, eval_freq:  int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn  # Function to create fresh env
        self.eval_freq = eval_freq
        self. history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            try:
                # Create a fresh evaluation environment each time
                eval_env = self.eval_env_fn()

                obs, _ = eval_env.reset()
                done = False
                max_steps = 1000  # Safety limit to prevent infinite loops
                step_count = 0

                while not done and step_count < max_steps:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    step_count += 1

                # Get metrics
                metrics = eval_env.get_episode_metrics()
                if metrics:  # Only add if we got valid metrics
                    metrics['timestep'] = self.n_calls
                    self.history.append(metrics)

                    if self.verbose:
                        print(f"\n[Eval @ {self.n_calls:,}] "
                              f"Return: {metrics. get('total_return', 0):.2%} | "
                              f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f} | "
                              f"MaxDD: {metrics.get('max_drawdown', 0):.2%}")
            except Exception as e:
                if self.verbose:
                    print(f"\n[Eval @ {self. n_calls:,}] Evaluation failed: {e}")

        return True

# ============================================
# FAST TRAINING SETUP
# ============================================

# Optimized parameters
WINDOW_SIZE = 10
EPISODE_LENGTH = 50  # Shorter episodes = faster training
TOTAL_TIMESTEPS = 30000  # Reduced but effective

def make_fast_train_env():
    env = PortfolioEnvFast(
        features_df=train_features,
        returns_df=train_returns,
        window_size=WINDOW_SIZE,
        episode_length=EPISODE_LENGTH,
        transaction_cost=0.001,
        random_start=True  # Important for learning diversity
    )
    return env

# Create vectorized environment
vec_train_env = DummyVecEnv([make_fast_train_env])

print("✅ Fast training environment created!")
print(f"   Window size: {WINDOW_SIZE}")
print(f"   Episode length: {EPISODE_LENGTH}")
print(f"   Training samples: {len(train_features)}")

# ============================================
# OPTIMIZED PPO TRAINING (NO CALLBACK)
# ============================================

print("\n" + "=" * 60)
print("TRAINING PPO AGENT (OPTIMIZED)")
print("=" * 60)

# Optimized PPO hyperparameters for faster convergence
model = PPO(
    "MlpPolicy",
    vec_train_env,
    learning_rate=1e-3,        # Higher LR for faster learning
    n_steps=512,               # Smaller buffer = more frequent updates
    batch_size=128,            # Larger batch for stability
    n_epochs=5,                # Fewer epochs per update
    gamma=0.95,                # Slightly lower discount
    gae_lambda=0.9,
    clip_range=0.2,
    ent_coef=0.02,             # More exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,                 # Show training progress
    seed=42,
    device="auto",
    policy_kwargs=dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])  # Smaller network = faster
    )
)

print(f"\nTraining for {TOTAL_TIMESTEPS: ,} timesteps...")
print("This should take 1-3 minutes.\n")

# Train WITHOUT callback (much faster)
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True
)

print("\n✅ Training complete!")
model.save("portfolio_ppo_fast")
print("Model saved!")

# ============================================
# QUICK EVALUATION AFTER TRAINING
# ============================================

print("\n" + "=" * 60)
print("EVALUATING TRAINED AGENT")
print("=" * 60)

# Create evaluation environment (full test period, no random start)
eval_env = PortfolioEnvFast(
    features_df=test_features,
    returns_df=test_returns,
    window_size=WINDOW_SIZE,
    episode_length=len(test_features) - WINDOW_SIZE - 1,  # Full test period
    transaction_cost=0.001,
    random_start=False  # Start from beginning
)

# Run evaluation
obs, _ = eval_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

# Get metrics
rl_metrics = eval_env.get_episode_metrics()
rl_returns = np.array(eval_env.returns_history)
rl_weights = np.array(eval_env.weights_history[1:])

print(f"\n📊 RL Agent Performance:")
print(f"   Total Return:     {rl_metrics['total_return']:.2%}")
print(f"   Annual Return:   {rl_metrics['annual_return']:.2%}")
print(f"   Sharpe Ratio:    {rl_metrics['sharpe_ratio']:.3f}")
print(f"   Max Drawdown:    {rl_metrics['max_drawdown']:.2%}")
print(f"   Win Rate:         {rl_metrics['win_rate']:.1%}")
print(f"   Trading Days:     {rl_metrics['n_trading_days']}")

# Plot training progress
# Check if 'callback' variable exists and has history before attempting to plot
if 'callback' in globals() and hasattr(callback, 'history') and len(callback.history) > 0:
    history_df = pd.DataFrame(callback.history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history_df['timestep'], history_df['total_return'], 'b-o', markersize=4)
    axes[0, 0].set_xlabel('Training Timesteps')
    axes[0, 0].set_ylabel('Total Return')
    axes[0, 0].set_title('Training Progress: Total Return')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    axes[0, 1].plot(history_df['timestep'], history_df['sharpe_ratio'], 'g-o', markersize=4)
    axes[0, 1].set_xlabel('Training Timesteps')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Training Progress: Sharpe Ratio')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history_df['timestep'], history_df['max_drawdown'], 'r-o', markersize=4)
    axes[1, 0].set_xlabel('Training Timesteps')
    axes[1, 0].set_ylabel('Max Drawdown')
    axes[1, 0].set_title('Training Progress: Max Drawdown')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    axes[1, 1].plot(history_df['timestep'], history_df['win_rate'], 'm-o', markersize=4)
    axes[1, 1].set_xlabel('Training Timesteps')
    axes[1, 1].set_ylabel('Win Rate')
    axes[1, 1].set_title('Training Progress: Win Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("Skipping training progress plots: 'callback' object not found or has no history.")
    print("To generate these plots, you need to enable the TrainingCallback during the PPO agent training phase (cell N0hBCu1IITAB).")


# ============================================
# BENCHMARK STRATEGIES
# ============================================

class BenchmarkStrategies:
    """Collection of benchmark strategies for comparison."""

    def __init__(self, returns_df: pd.DataFrame):
        self.returns_df = returns_df
        self.n_assets = returns_df.shape[1]

    def equal_weight(self) -> Tuple[np.ndarray, np.ndarray]:
        """Equal-weight portfolio (1/N strategy)."""
        n_days = len(self.returns_df)
        weights = np.ones((n_days, self.n_assets)) / self.n_assets
        portfolio_returns = (self.returns_df.values * weights).sum(axis=1)
        return portfolio_returns, weights

    def risk_parity(self, lookback:  int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Risk parity portfolio (inverse volatility weighting)."""
        returns = self.returns_df.values
        n_days = len(returns)

        portfolio_returns = np.zeros(n_days)
        weights_history = np.zeros((n_days, self.n_assets))

        # Initial period with equal weights
        weights_history[: lookback] = 1.0 / self.n_assets
        portfolio_returns[:lookback] = (returns[:lookback] * weights_history[: lookback]).sum(axis=1)

        for t in range(lookback, n_days):
            hist_returns = returns[t-lookback:t]
            volatilities = hist_returns.std(axis=0)

            # Inverse volatility weights
            inv_vol = 1.0 / (volatilities + 1e-8)
            weights = inv_vol / inv_vol.sum()

            weights_history[t] = weights
            portfolio_returns[t] = np.dot(weights, returns[t])

        return portfolio_returns, weights_history

    def momentum(self, lookback: int = 60, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Momentum strategy - invest in top performing assets."""
        returns = self. returns_df.values
        n_days = len(returns)

        portfolio_returns = np.zeros(n_days)
        weights_history = np.zeros((n_days, self.n_assets))

        # Initial period
        weights_history[: lookback] = 1.0 / self.n_assets
        portfolio_returns[:lookback] = (returns[:lookback] * weights_history[:lookback]).sum(axis=1)

        for t in range(lookback, n_days):
            # Calculate momentum (cumulative return over lookback)
            hist_returns = returns[t-lookback:t]
            momentum = (1 + hist_returns).prod(axis=0) - 1

            # Select top k assets
            top_indices = np.argsort(momentum)[-top_k:]

            # Equal weight among top k
            weights = np.zeros(self.n_assets)
            weights[top_indices] = 1.0 / top_k

            weights_history[t] = weights
            portfolio_returns[t] = np.dot(weights, returns[t])

        return portfolio_returns, weights_history

    def minimum_variance(self, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Minimum variance portfolio."""
        returns = self.returns_df.values
        n_days = len(returns)

        portfolio_returns = np.zeros(n_days)
        weights_history = np.zeros((n_days, self.n_assets))

        # Initial period
        weights_history[:lookback] = 1.0 / self.n_assets
        portfolio_returns[:lookback] = (returns[:lookback] * weights_history[:lookback]).sum(axis=1)

        for t in range(lookback, n_days):
            hist_returns = returns[t-lookback: t]

            try:
                cov = np.cov(hist_returns. T)
                cov_inv = np.linalg.inv(cov + np.eye(self.n_assets) * 1e-6)
                ones = np.ones(self.n_assets)
                weights = cov_inv @ ones / (ones @ cov_inv @ ones)
                weights = np.maximum(weights, 0)  # Long only
                weights = weights / (weights.sum() + 1e-8)
            except:
                weights = np.ones(self.n_assets) / self.n_assets

            weights_history[t] = weights
            portfolio_returns[t] = np.dot(weights, returns[t])

        return portfolio_returns, weights_history

# ============================================
# PERFORMANCE METRICS CALCULATOR
# ============================================

def calculate_metrics(returns:  np.ndarray, risk_free_rate: float = 0.02) -> dict:
    """Calculate comprehensive performance metrics."""

    # Cumulative returns
    cumulative = (1 + returns).cumprod()
    total_return = cumulative[-1] - 1

    # Annualized return
    n_days = len(returns)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    # Volatility
    volatility = np. std(returns) * np.sqrt(252)

    # Downside volatility
    negative_returns = returns[returns < 0]
    downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0

    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else 0

    # Sortino ratio
    sortino = excess_return / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_drawdown = drawdown.max()

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown > 0 else 0

    # Win rate
    win_rate = np.mean(returns > 0)

    return {
        'total_return': total_return,
        'annual_return':  annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'cumulative_returns': cumulative
    }

# ============================================
# RUN BACKTESTS
# ============================================

print("=" * 60)
print("RUNNING BACKTESTS")
print("=" * 60)

# 1. RL Agent Backtest
print("\n1. Running RL Agent backtest...")
rl_env = PortfolioEnvFast(
    features_df=test_features,
    returns_df=test_returns,
    window_size=WINDOW_SIZE, # Changed from 50 to WINDOW_SIZE (20) for consistency with training
    transaction_cost=0.001
)

obs, _ = rl_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = rl_env.step(action)
    done = terminated or truncated

rl_returns = np.array(rl_env.returns_history)
rl_weights = np.array(rl_env.weights_history[1:])  # Skip initial weights
rl_metrics = calculate_metrics(rl_returns)
print(f"   RL Agent - Total Return: {rl_metrics['total_return']:.2%}, Sharpe: {rl_metrics['sharpe_ratio']:.3f}")

# 2. Benchmark Strategies
benchmarks = BenchmarkStrategies(test_returns)

print("\n2. Running benchmark strategies...")

# Equal Weight
ew_returns, ew_weights = benchmarks.equal_weight()
ew_metrics = calculate_metrics(ew_returns)
print(f"   Equal Weight - Total Return: {ew_metrics['total_return']:.2%}, Sharpe: {ew_metrics['sharpe_ratio']:.3f}")

# Risk Parity
rp_returns, rp_weights = benchmarks.risk_parity()
rp_metrics = calculate_metrics(rp_returns)
print(f"   Risk Parity - Total Return:  {rp_metrics['total_return']:.2%}, Sharpe: {rp_metrics['sharpe_ratio']:.3f}")

# Momentum
mom_returns, mom_weights = benchmarks.momentum()
mom_metrics = calculate_metrics(mom_returns)
print(f"   Momentum - Total Return: {mom_metrics['total_return']:.2%}, Sharpe: {mom_metrics['sharpe_ratio']:.3f}")

# Minimum Variance
mv_returns, mv_weights = benchmarks.minimum_variance()
mv_metrics = calculate_metrics(mv_returns)
print(f"   Min Variance - Total Return: {mv_metrics['total_return']:.2%}, Sharpe: {mv_metrics['sharpe_ratio']:.3f}")

# SPY Buy & Hold
spy_returns_test = test_spy_returns.values[: len(rl_returns)]
spy_metrics = calculate_metrics(spy_returns_test)
print(f"   SPY (Buy & Hold) - Total Return: {spy_metrics['total_return']:.2%}, Sharpe: {spy_metrics['sharpe_ratio']:.3f}")

# ============================================
# CREATE COMPARISON TABLE
# ============================================

print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

results = {
    'RL Agent (PPO)': rl_metrics,
    'Equal Weight': ew_metrics,
    'Risk Parity': rp_metrics,
    'Momentum': mom_metrics,
    'Min Variance': mv_metrics,
    'SPY (Buy & Hold)': spy_metrics
}

comparison_data = []
for name, metrics in results. items():
    comparison_data. append({
        'Strategy': name,
        'Total Return': f"{metrics['total_return']:.2%}",
        'Annual Return': f"{metrics['annual_return']:.2%}",
        'Volatility': f"{metrics['volatility']:.2%}",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
        'Sortino Ratio': f"{metrics['sortino_ratio']:.3f}",
        'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
        'Win Rate':  f"{metrics['win_rate']:.1%}"
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.set_index('Strategy')
print("\n")
print(comparison_df.to_string())

# Save comparison
comparison_df.to_csv('backtest_comparison.csv')
print("\nComparison saved to 'backtest_comparison.csv'")

# ============================================
# PORTFOLIO VALUE COMPARISON
# ============================================

initial_capital = 100000

fig, ax = plt.subplots(figsize=(14, 8))

# Calculate portfolio values
dates = test_features.index[:len(rl_returns)]

strategies = {
    'RL Agent (PPO)': rl_metrics['cumulative_returns'],
    'Equal Weight': ew_metrics['cumulative_returns'][:len(rl_returns)],
    'Risk Parity': rp_metrics['cumulative_returns'][:len(rl_returns)],
    'Momentum': mom_metrics['cumulative_returns'][:len(rl_returns)],
    'Min Variance': mv_metrics['cumulative_returns'][:len(rl_returns)],
    'SPY (Buy & Hold)': spy_metrics['cumulative_returns']
}

colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

for (name, cum_returns), color in zip(strategies.items(), colors):
    portfolio_values = initial_capital * cum_returns
    linewidth = 3 if 'RL' in name else 1.5
    linestyle = '-' if 'RL' in name else '--'
    ax.plot(dates[:len(portfolio_values)], portfolio_values,
            label=name, color=color, linewidth=linewidth, linestyle=linestyle)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Value ($)', fontsize=12)
ax.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('portfolio_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# DRAWDOWN COMPARISON
# ============================================

fig, ax = plt.subplots(figsize=(14, 6))

for (name, cum_returns), color in zip(strategies.items(), colors):
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (peak - cum_returns) / peak * 100

    linewidth = 2 if 'RL' in name else 1
    ax.fill_between(dates[:len(drawdown)], 0, -drawdown,
                    alpha=0.3 if 'RL' not in name else 0.5,
                    label=name, color=color)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.set_title('Portfolio Drawdowns', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('drawdowns.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# METRICS BAR CHART
# ============================================

metrics_to_plot = ['sharpe_ratio', 'sortino_ratio', 'annual_return', 'max_drawdown']
metric_labels = ['Sharpe Ratio', 'Sortino Ratio', 'Annual Return', 'Max Drawdown']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

strategy_names = list(results.keys())

for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    ax = axes[idx]

    values = [results[name][metric] for name in strategy_names]

    bars = ax.bar(strategy_names, values, color=colors, alpha=0.8, edgecolor='black')

    # Highlight best
    if metric == 'max_drawdown':
        best_idx = np.argmin(np.abs(values))
    else:
        best_idx = np.argmax(values)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    if metric in ['annual_return', 'max_drawdown']:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

plt.suptitle('Strategy Metrics Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# RL AGENT PORTFOLIO WEIGHTS OVER TIME
# ============================================

fig, ax = plt.subplots(figsize=(14, 8))

weights_df = pd.DataFrame(rl_weights, columns=asset_names, index=dates[: len(rl_weights)])

ax.stackplot(weights_df.index, weights_df.T, labels=asset_names, alpha=0.8)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Weight', fontsize=12)
ax.set_title('RL Agent - Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_weights.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# AVERAGE WEIGHTS COMPARISON
# ============================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# RL Agent average weights
ax1 = axes[0]
avg_rl_weights = rl_weights.mean(axis=0)
ax1.pie(avg_rl_weights, labels=asset_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
ax1.set_title('RL Agent\nAverage Allocation', fontsize=12, fontweight='bold')

# Equal weight
ax2 = axes[1]
equal_weights = np.ones(n_assets) / n_assets
ax2.pie(equal_weights, labels=asset_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
ax2.set_title('Equal Weight\nAllocation', fontsize=12, fontweight='bold')

# Risk Parity average weights
ax3 = axes[2]
avg_rp_weights = rp_weights.mean(axis=0)
ax3.pie(avg_rp_weights, labels=asset_names, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
ax3.set_title('Risk Parity\nAverage Allocation', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('average_weights.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# PERMUTATION IMPORTANCE (SIMPLIFIED)
# ============================================

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

def evaluate_agent(model, features, returns, n_runs:  int = 3) -> float:
    """Evaluate agent and return average Sharpe ratio."""
    sharpe_ratios = []

    for _ in range(n_runs):
        env = PortfolioEnvFast(features_df=features, returns_df=returns, window_size=WINDOW_SIZE)
        obs, _ = env. reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        metrics = env.get_episode_metrics()
        sharpe_ratios.append(metrics['sharpe_ratio'])

    return np.mean(sharpe_ratios)

# Baseline performance
print("\nCalculating baseline performance...")
baseline_sharpe = evaluate_agent(model, test_features, test_returns)
print(f"Baseline Sharpe Ratio: {baseline_sharpe:.4f}")

# Permutation importance for each feature (asset)
print("\nCalculating permutation importance...")
importance_scores = {}

for col in test_features.columns:
    # Create permuted features
    permuted_features = test_features.copy()
    permuted_features[col] = np.random.permutation(permuted_features[col].values)

    # Evaluate with permuted feature
    permuted_sharpe = evaluate_agent(model, permuted_features, test_returns, n_runs=1)

    # Importance = drop in performance
    importance = baseline_sharpe - permuted_sharpe
    importance_scores[col] = importance
    print(f"  {col}: importance = {importance:.4f}")

# Visualize feature importance
importance_df = pd.DataFrame({
    'Feature': importance_scores.keys(),
    'Importance':  importance_scores.values()
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['green' if x > 0 else 'red' for x in importance_df['Importance']]
ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Importance (Sharpe Drop when Permuted)', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Feature Importance Analysis\n(Higher = More Important)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# GENERATE SUMMARY REPORT
# ============================================

print("\n" + "=" * 70)
print("                    PORTFOLIO OPTIMIZATION - SUMMARY REPORT")
print("=" * 70)

print(f"""
📊 DATA SUMMARY
{'─' * 50}
• Assets Traded: {', '.join(asset_names)}
• Training Period: {train_features.index[0]. date()} to {train_features.index[-1].date()}
• Testing Period: {test_features.index[0].date()} to {test_features.index[-1]. date()}
• Training Samples: {len(train_features):,}
• Testing Samples:  {len(test_features):,}

🤖 RL AGENT CONFIGURATION
{'─' * 50}
• Algorithm: PPO (Proximal Policy Optimization)
• Window Size: 50 days
• Transaction Cost: 0.1%
• Training Timesteps: 100,000

📈 RL AGENT PERFORMANCE (Test Period)
{'─' * 50}
• Total Return: {rl_metrics['total_return']:.2%}
• Annualized Return: {rl_metrics['annual_return']:.2%}
• Volatility:  {rl_metrics['volatility']:.2%}
• Sharpe Ratio: {rl_metrics['sharpe_ratio']:.3f}
• Sortino Ratio: {rl_metrics['sortino_ratio']:.3f}
• Max Drawdown: {rl_metrics['max_drawdown']:.2%}
• Win Rate: {rl_metrics['win_rate']:.1%}

🏆 STRATEGY RANKINGS (by Sharpe Ratio)
{'─' * 50}""")

# Rank by Sharpe
sharpe_ranking = sorted(results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
for rank, (name, metrics) in enumerate(sharpe_ranking, 1):
    print(f"  {rank}. {name}:  {metrics['sharpe_ratio']:.3f}")

print(f"""
📁 SAVED FILES
{'─' * 50}
• Model: portfolio_ppo_model.zip
• Comparison: backtest_comparison.csv
• Charts: price_analysis.png, training_progress.png,
          portfolio_comparison.png, drawdowns. png,
          metrics_comparison.png, rl_weights.png,
          average_weights.png, feature_importance.png
""")

print("=" * 70)
print("                         END OF REPORT")
print("=" * 70)

# Save summary to file
with open('summary_report.txt', 'w') as f:
    f.write("PORTFOLIO OPTIMIZATION USING DEEP REINFORCEMENT LEARNING\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("PERFORMANCE COMPARISON\n")
    f.write("-" * 60 + "\n")
    f.write(comparison_df.to_string())
    f.write("\n\n")
    f.write("STRATEGY RANKINGS (by Sharpe Ratio)\n")
    f.write("-" * 60 + "\n")
    for rank, (name, metrics) in enumerate(sharpe_ranking, 1):
        f.write(f"{rank}. {name}: {metrics['sharpe_ratio']:.3f}\n")

print("\nSummary report saved to 'summary_report.txt'")
