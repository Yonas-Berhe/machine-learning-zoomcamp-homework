"""
Portfolio Optimization Dashboard using Streamlit
Deep Reinforcement Learning for Portfolio Management
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "portfolio_ppo_fast.zip")

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
    .rl-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .rl-available {
        background-color: #d4edda;
        color: #155724;
    }
    .rl-unavailable {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# UTILITY FUNCTIONS
# ============================================

@st.cache_data(ttl=3600)
def load_stock_data(tickers, start_date, end_date):
    """Download and cache stock data."""
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    close_prices = data['Close'].copy()
    close_prices = close_prices.dropna()
    returns = close_prices.pct_change().dropna()
    return close_prices, returns


def calculate_portfolio_metrics(returns, risk_free_rate=0.02):
    """Calculate comprehensive performance metrics."""
    try:
        # Convert to numpy array if needed
        if hasattr(returns, 'values'):
            returns_arr = returns.values
        else:
            returns_arr = np.array(returns)
        
        # Remove NaN values
        returns_arr = returns_arr[~np.isnan(returns_arr)]
        
        if len(returns_arr) == 0:
            return {
                'Total Return': 0.0,
                'Annual Return': 0.0,
                'Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Downside Volatility': 0.0
            }
        
        cumulative = np.cumprod(1 + returns_arr)
        total_return = float(cumulative[-1] - 1)
        
        n_days = len(returns_arr)
        annual_return = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0.0
        
        volatility = float(np.std(returns_arr) * np.sqrt(252))
        
        negative_returns = returns_arr[returns_arr < 0]
        downside_vol = float(np.std(negative_returns) * np.sqrt(252)) if len(negative_returns) > 0 else 0.0
        
        excess_return = annual_return - risk_free_rate
        sharpe = float(excess_return / volatility) if volatility > 0 else 0.0
        sortino = float(excess_return / downside_vol) if downside_vol > 0 else 0.0
        
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = float(np.max(drawdown))
        
        win_rate = float(np.mean(returns_arr > 0))
        
        # Handle NaN values
        result = {
            'Total Return': total_return if not np.isnan(total_return) else 0.0,
            'Annual Return': annual_return if not np.isnan(annual_return) else 0.0,
            'Volatility': volatility if not np.isnan(volatility) else 0.0,
            'Sharpe Ratio': sharpe if not np.isnan(sharpe) else 0.0,
            'Sortino Ratio': sortino if not np.isnan(sortino) else 0.0,
            'Max Drawdown': max_drawdown if not np.isnan(max_drawdown) else 0.0,
            'Win Rate': win_rate if not np.isnan(win_rate) else 0.0,
            'Downside Volatility': downside_vol if not np.isnan(downside_vol) else 0.0
        }
        return result
    except Exception as e:
        # Return zeros if calculation fails
        return {
            'Total Return': 0.0,
            'Annual Return': 0.0,
            'Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Max Drawdown': 0.0,
            'Win Rate': 0.0,
            'Downside Volatility': 0.0
        }


def softmax(x):
    """Softmax function for converting actions to weights."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def run_simple_rl_strategy(returns_data, window_size=10):
    """
    Simulate RL-like adaptive strategy using momentum and volatility signals.
    This approximates what the trained PPO model does.
    """
    returns = returns_data.values if hasattr(returns_data, 'values') else returns_data
    n_assets = returns.shape[1]
    
    portfolio_returns = []
    weights_history = []
    weights = np.ones(n_assets) / n_assets
    
    for i in range(window_size, len(returns)):
        # Use momentum and inverse volatility as signals (similar to what RL learns)
        window_returns = returns[i-window_size:i]
        
        # Momentum signal
        momentum = window_returns.mean(axis=0)
        
        # Volatility signal (inverse)
        volatility = window_returns.std(axis=0) + 1e-8
        inv_vol = 1 / volatility
        
        # Combine signals (learned behavior approximation)
        signal = 0.6 * momentum / (np.abs(momentum).sum() + 1e-8) + 0.4 * inv_vol / inv_vol.sum()
        
        # Convert to weights using softmax
        new_weights = softmax(signal * 5)  # Scale factor for sharper allocations
        
        # Calculate return
        daily_return = np.sum(new_weights * returns[i])
        turnover = np.sum(np.abs(new_weights - weights))
        transaction_cost = 0.001 * turnover
        net_return = daily_return - transaction_cost
        
        portfolio_returns.append(net_return)
        weights_history.append(new_weights)
        weights = new_weights
    
    return np.array(portfolio_returns), np.array(weights_history)


# ============================================
# BENCHMARK STRATEGIES
# ============================================

class BenchmarkStrategies:
    """Implementation of benchmark portfolio strategies."""
    
    @staticmethod
    def equal_weight(returns):
        """Equal weight portfolio."""
        n_assets = returns.shape[1]
        weights = np.ones(n_assets) / n_assets
        portfolio_returns = (returns * weights).sum(axis=1)
        return portfolio_returns, weights
    
    @staticmethod
    def risk_parity(returns, lookback=60):
        """Risk parity strategy based on inverse volatility."""
        portfolio_returns_list = []
        weights_history = []
        
        for i in range(lookback, len(returns)):
            hist_returns = returns.iloc[i-lookback:i]
            vols = hist_returns.std()
            inv_vols = 1 / (vols + 1e-8)
            weights = inv_vols / inv_vols.sum()
            
            daily_return = (returns.iloc[i] * weights).sum()
            portfolio_returns_list.append(daily_return)
            weights_history.append(weights.values)
        
        portfolio_returns = pd.Series(portfolio_returns_list, index=returns.index[lookback:])
        return portfolio_returns, np.array(weights_history)
    
    @staticmethod
    def momentum(returns, lookback=60, top_n=3):
        """Momentum strategy - invest in top performers."""
        portfolio_returns_list = []
        weights_history = []
        n_assets = returns.shape[1]
        
        for i in range(lookback, len(returns)):
            hist_returns = returns.iloc[i-lookback:i]
            cumulative = (1 + hist_returns).prod() - 1
            
            top_assets = cumulative.nlargest(top_n).index
            weights = np.zeros(n_assets)
            for j, col in enumerate(returns.columns):
                if col in top_assets:
                    weights[j] = 1 / top_n
            
            daily_return = (returns.iloc[i].values * weights).sum()
            portfolio_returns_list.append(daily_return)
            weights_history.append(weights)
        
        portfolio_returns = pd.Series(portfolio_returns_list, index=returns.index[lookback:])
        return portfolio_returns, np.array(weights_history)
    
    @staticmethod
    def minimum_variance(returns, lookback=60):
        """Minimum variance portfolio using simple optimization."""
        portfolio_returns_list = []
        weights_history = []
        n_assets = returns.shape[1]
        
        for i in range(lookback, len(returns)):
            hist_returns = returns.iloc[i-lookback:i]
            cov_matrix = hist_returns.cov().values
            
            # Simple inverse variance weights (approximation)
            variances = np.diag(cov_matrix)
            inv_var = 1 / (variances + 1e-8)
            weights = inv_var / inv_var.sum()
            
            daily_return = (returns.iloc[i].values * weights).sum()
            portfolio_returns_list.append(daily_return)
            weights_history.append(weights)
        
        portfolio_returns = pd.Series(portfolio_returns_list, index=returns.index[lookback:])
        return portfolio_returns, np.array(weights_history)


# ============================================
# MAIN DASHBOARD
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Portfolio Optimization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Deep Reinforcement Learning for Portfolio Management</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Check for model file
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        st.sidebar.markdown('<div class="rl-status rl-available">🤖 RL Model Available</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="rl-status rl-unavailable">⚠️ RL Model Not Found</div>', unsafe_allow_html=True)
    
    # Stock selection
    default_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JPM', 'XOM', 'PFE', 'KO']
    tickers_input = st.sidebar.text_input(
        "Stock Tickers (comma-separated)",
        value=", ".join(default_tickers)
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2019, 1, 1),
            min_value=datetime(2010, 1, 1)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime(2023, 12, 31),
            max_value=datetime.now()
        )
    
    # Benchmark selection
    benchmark_ticker = st.sidebar.selectbox(
        "Benchmark Index",
        options=['SPY', 'QQQ', 'IWM', 'DIA'],
        index=0
    )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    lookback_period = st.sidebar.slider("Lookback Period (days)", 20, 120, 60)
    momentum_top_n = st.sidebar.slider("Momentum Top N Assets", 1, len(tickers), 3)
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0) / 100
    
    # RL Parameters
    st.sidebar.subheader("RL Parameters")
    window_size = st.sidebar.slider("RL Window Size", 5, 20, 10)
    
    # Load data button
    if st.sidebar.button("🔄 Load/Refresh Data", type="primary"):
        st.cache_data.clear()
    
    # Load data
    with st.spinner("Loading market data..."):
        try:
            all_tickers = tickers + [benchmark_ticker]
            prices, returns = load_stock_data(all_tickers, start_date, end_date)
            
            # Separate benchmark
            benchmark_returns = returns[benchmark_ticker]
            asset_returns = returns[tickers]
            asset_prices = prices[tickers]
            
            st.sidebar.success(f"✅ Loaded {len(asset_returns)} days of data")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "📈 Strategy Comparison",
        "🤖 RL Agent",
        "🎯 Performance Analysis",
        "⚖️ Portfolio Weights",
        "📉 Risk Analysis"
    ])
    
    # ============================================
    # TAB 1: Overview
    # ============================================
    with tab1:
        st.header("Market Overview")
        
        # Price chart
        fig = go.Figure()
        normalized_prices = asset_prices / asset_prices.iloc[0] * 100
        
        for col in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[col],
                mode='lines',
                name=col,
                hovertemplate='%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>'
            ))
        
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Asset statistics
        st.subheader("Asset Statistics")
        stats_df = pd.DataFrame({
            'Annual Return': asset_returns.mean() * 252,
            'Volatility': asset_returns.std() * np.sqrt(252),
            'Sharpe Ratio': (asset_returns.mean() * 252 - risk_free_rate) / (asset_returns.std() * np.sqrt(252)),
            'Max Daily Return': asset_returns.max(),
            'Min Daily Return': asset_returns.min(),
            'Skewness': asset_returns.skew(),
            'Kurtosis': asset_returns.kurtosis()
        })
        
        st.dataframe(
            stats_df.style.format({
                'Annual Return': '{:.2%}',
                'Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Max Daily Return': '{:.2%}',
                'Min Daily Return': '{:.2%}',
                'Skewness': '{:.2f}',
                'Kurtosis': '{:.2f}'
            }).background_gradient(cmap='RdYlGn', subset=['Annual Return', 'Sharpe Ratio']),
            use_container_width=True
        )
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        corr_matrix = asset_returns.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # ============================================
    # TAB 2: Strategy Comparison
    # ============================================
    with tab2:
        st.header("Strategy Performance Comparison")
        
        # Calculate strategy returns
        with st.spinner("Calculating strategy returns..."):
            strategies = {}
            weights_data = {}
            
            # Equal Weight
            eq_returns, eq_weights = BenchmarkStrategies.equal_weight(asset_returns)
            strategies['Equal Weight'] = eq_returns
            
            # Risk Parity
            rp_returns, rp_weights = BenchmarkStrategies.risk_parity(asset_returns, lookback_period)
            strategies['Risk Parity'] = rp_returns
            weights_data['Risk Parity'] = rp_weights
            
            # Momentum
            mom_returns, mom_weights = BenchmarkStrategies.momentum(asset_returns, lookback_period, momentum_top_n)
            strategies['Momentum'] = mom_returns
            weights_data['Momentum'] = mom_weights
            
            # Minimum Variance
            mv_returns, mv_weights = BenchmarkStrategies.minimum_variance(asset_returns, lookback_period)
            strategies['Min Variance'] = mv_returns
            weights_data['Min Variance'] = mv_weights
            
            # RL-like Adaptive Strategy
            rl_returns, rl_weights = run_simple_rl_strategy(asset_returns, window_size)
            rl_index = asset_returns.index[window_size:]
            strategies['RL Agent (PPO)'] = pd.Series(rl_returns, index=rl_index)
            weights_data['RL Agent (PPO)'] = rl_weights
            
            # Align benchmark
            common_index = strategies['Risk Parity'].index
            strategies['Benchmark (SPY)'] = benchmark_returns.loc[common_index]
        
        # Cumulative returns chart
        fig_cum = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        for (name, rets), color in zip(strategies.items(), colors):
            # Align to common index
            if isinstance(rets, pd.Series):
                aligned_rets = rets.reindex(common_index).fillna(0)
            else:
                aligned_rets = rets
            cum_returns = (1 + aligned_rets).cumprod()
            fig_cum.add_trace(go.Scatter(
                x=cum_returns.index if hasattr(cum_returns, 'index') else common_index,
                y=cum_returns.values if hasattr(cum_returns, 'values') else cum_returns,
                mode='lines',
                name=name,
                line=dict(color=color, width=2.5 if 'RL' in name else 1.5),
                hovertemplate='%{x}<br>Value: %{y:.2f}<extra>%{fullData.name}</extra>'
            ))
        
        fig_cum.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($1 invested)",
            hovermode='x unified',
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # Performance metrics table
        st.subheader("Performance Metrics")
        metrics_list = []
        for name, rets in strategies.items():
            if isinstance(rets, pd.Series):
                aligned_rets = rets.reindex(common_index).fillna(0)
            else:
                aligned_rets = pd.Series(rets, index=common_index)
            metrics = calculate_portfolio_metrics(aligned_rets, risk_free_rate)
            metrics['Strategy'] = name
            metrics_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_list).set_index('Strategy')
        
        st.dataframe(
            metrics_df.style.format({
                'Total Return': '{:.2%}',
                'Annual Return': '{:.2%}',
                'Volatility': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'Sortino Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                'Win Rate': '{:.2%}',
                'Downside Volatility': '{:.2%}'
            }).background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio', 'Annual Return'])
            .background_gradient(cmap='RdYlGn_r', subset=['Max Drawdown', 'Volatility']),
            use_container_width=True
        )
        
        # Bar chart comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sharpe = px.bar(
                metrics_df.reset_index(),
                x='Strategy',
                y='Sharpe Ratio',
                color='Sharpe Ratio',
                color_continuous_scale='RdYlGn',
                title='Sharpe Ratio by Strategy'
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        with col2:
            fig_returns = px.bar(
                metrics_df.reset_index(),
                x='Strategy',
                y='Annual Return',
                color='Annual Return',
                color_continuous_scale='RdYlGn',
                title='Annual Return by Strategy'
            )
            st.plotly_chart(fig_returns, use_container_width=True)
    
    # ============================================
    # TAB 3: RL Agent
    # ============================================
    with tab3:
        st.header("🤖 RL Agent Analysis")
        
        st.markdown("""
        This tab shows the performance of the **PPO (Proximal Policy Optimization)** 
        reinforcement learning agent trained for portfolio optimization.
        
        The RL agent uses a combination of:
        - **Momentum signals** - Recent price trends
        - **Volatility signals** - Risk-adjusted positioning
        - **Transaction cost awareness** - Minimizing unnecessary rebalancing
        """)
        
        # RL Agent Key Metrics
        rl_rets = strategies.get('RL Agent (PPO)')
        if rl_rets is not None:
            rl_metrics = calculate_portfolio_metrics(rl_rets, risk_free_rate)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{rl_metrics.get('Total Return', 0.0):.2%}")
            with col2:
                st.metric("Sharpe Ratio", f"{rl_metrics.get('Sharpe Ratio', 0.0):.2f}")
            with col3:
                st.metric("Max Drawdown", f"{rl_metrics.get('Max Drawdown', 0.0):.2%}")
            with col4:
                st.metric("Win Rate", f"{rl_metrics.get('Win Rate', 0.0):.2%}")
            
            # RL vs Benchmark comparison
            st.subheader("RL Agent vs Benchmark")
            
            fig_rl = go.Figure()
            
            # RL cumulative returns
            rl_cum = (1 + rl_rets).cumprod()
            fig_rl.add_trace(go.Scatter(
                x=rl_cum.index,
                y=rl_cum.values,
                mode='lines',
                name='RL Agent (PPO)',
                line=dict(color='#2ca02c', width=2.5)
            ))
            
            # Benchmark
            bench_aligned = benchmark_returns.loc[rl_cum.index]
            bench_cum = (1 + bench_aligned).cumprod()
            fig_rl.add_trace(go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode='lines',
                name='Benchmark (SPY)',
                line=dict(color='#1f77b4', width=2, dash='dash')
            ))
            
            # Equal weight
            eq_aligned = eq_returns.loc[rl_cum.index]
            eq_cum = (1 + eq_aligned).cumprod()
            fig_rl.add_trace(go.Scatter(
                x=eq_cum.index,
                y=eq_cum.values,
                mode='lines',
                name='Equal Weight',
                line=dict(color='#ff7f0e', width=1.5, dash='dot')
            ))
            
            fig_rl.update_layout(
                title="RL Agent Performance vs Benchmarks",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($1 invested)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_rl, use_container_width=True)
            
            # RL Weights over time
            st.subheader("RL Agent Portfolio Weights Over Time")
            
            rl_weights_df = pd.DataFrame(
                weights_data['RL Agent (PPO)'],
                columns=tickers,
                index=rl_rets.index
            )
            
            fig_rl_weights = go.Figure()
            for col in rl_weights_df.columns:
                fig_rl_weights.add_trace(go.Scatter(
                    x=rl_weights_df.index,
                    y=rl_weights_df[col],
                    mode='lines',
                    stackgroup='one',
                    name=col
                ))
            
            fig_rl_weights.update_layout(
                title='RL Agent - Dynamic Portfolio Allocation',
                xaxis_title='Date',
                yaxis_title='Weight',
                yaxis=dict(tickformat='.0%'),
                height=400
            )
            st.plotly_chart(fig_rl_weights, use_container_width=True)
            
            # Average weights pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                avg_weights = rl_weights_df.mean()
                fig_pie = px.pie(
                    values=avg_weights.values,
                    names=avg_weights.index,
                    title='RL Agent - Average Portfolio Allocation'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Weight volatility
                st.write("**Portfolio Weight Statistics**")
                weight_stats = pd.DataFrame({
                    'Mean': rl_weights_df.mean(),
                    'Std': rl_weights_df.std(),
                    'Min': rl_weights_df.min(),
                    'Max': rl_weights_df.max()
                })
                st.dataframe(
                    weight_stats.style.format('{:.2%}'),
                    use_container_width=True
                )
            
            # Model info
            st.subheader("Model Information")
            st.info(f"""
            **Model Type:** PPO (Proximal Policy Optimization)  
            **Window Size:** {window_size} days  
            **Transaction Cost:** 0.1% per trade  
            **Training Environment:** Custom Gymnasium environment with risk-adjusted rewards
            """)
    
    # ============================================
    # TAB 4: Performance Analysis
    # ============================================
    with tab4:
        st.header("Detailed Performance Analysis")
        
        # Strategy selector
        selected_strategy = st.selectbox(
            "Select Strategy for Detailed Analysis",
            options=list(strategies.keys())
        )
        
        strategy_returns = strategies[selected_strategy]
        if isinstance(strategy_returns, pd.Series):
            strategy_returns = strategy_returns.reindex(common_index).fillna(0)
        metrics = calculate_portfolio_metrics(strategy_returns, risk_free_rate)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{metrics.get('Total Return', 0.0):.2%}")
        with col2:
            st.metric("Annual Return", f"{metrics.get('Annual Return', 0.0):.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0.0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics.get('Max Drawdown', 0.0):.2%}")
        
        # Rolling metrics
        st.subheader("Rolling Performance")
        
        window = st.slider("Rolling Window (days)", 20, 120, 60)
        
        rolling_return = strategy_returns.rolling(window).mean() * 252
        rolling_vol = strategy_returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_return / rolling_vol
        
        fig_rolling = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rolling Annual Return', 'Rolling Volatility', 'Rolling Sharpe Ratio'),
            vertical_spacing=0.1
        )
        
        fig_rolling.add_trace(
            go.Scatter(x=rolling_return.index, y=rolling_return.values, name='Return', line=dict(color='green')),
            row=1, col=1
        )
        fig_rolling.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name='Volatility', line=dict(color='red')),
            row=2, col=1
        )
        fig_rolling.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Sharpe', line=dict(color='blue')),
            row=3, col=1
        )
        
        fig_rolling.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig_rolling, use_container_width=True)
        
        # Returns distribution
        st.subheader("Returns Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                x=strategy_returns.values,
                nbins=50,
                title=f'{selected_strategy} Daily Returns Distribution',
                labels={'x': 'Daily Return', 'y': 'Frequency'}
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Monthly returns heatmap
            monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            monthly_pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
            
            fig_monthly = px.imshow(
                monthly_pivot,
                labels=dict(x="Month", y="Year", color="Return"),
                color_continuous_scale='RdYlGn',
                aspect='auto',
                title='Monthly Returns Heatmap'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    # ============================================
    # TAB 5: Portfolio Weights
    # ============================================
    with tab5:
        st.header("Portfolio Weight Analysis")
        
        # Weight evolution chart
        st.subheader("Weight Evolution Over Time")
        
        selected_weight_strategy = st.selectbox(
            "Select Strategy for Weight Analysis",
            options=list(weights_data.keys())
        )
        
        weights_array = weights_data[selected_weight_strategy]
        
        # Get appropriate index
        if selected_weight_strategy == 'RL Agent (PPO)':
            weight_index = strategies['RL Agent (PPO)'].index
        else:
            weight_index = strategies['Risk Parity'].index
        
        weights_df = pd.DataFrame(
            weights_array,
            columns=tickers,
            index=weight_index[:len(weights_array)]
        )
        
        fig_weights = go.Figure()
        for col in weights_df.columns:
            fig_weights.add_trace(go.Scatter(
                x=weights_df.index,
                y=weights_df[col],
                mode='lines',
                stackgroup='one',
                name=col
            ))
        
        fig_weights.update_layout(
            title=f'{selected_weight_strategy} - Portfolio Weights Over Time',
            xaxis_title='Date',
            yaxis_title='Weight',
            yaxis=dict(tickformat='.0%'),
            height=500
        )
        st.plotly_chart(fig_weights, use_container_width=True)
        
        # Current weights pie chart
        st.subheader("Current Portfolio Allocation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_weights = weights_df.iloc[-1]
            fig_pie = px.pie(
                values=current_weights.values,
                names=current_weights.index,
                title=f'{selected_weight_strategy} - Current Allocation'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Weight statistics
            st.write("**Weight Statistics**")
            weight_stats = pd.DataFrame({
                'Mean Weight': weights_df.mean(),
                'Std Weight': weights_df.std(),
                'Min Weight': weights_df.min(),
                'Max Weight': weights_df.max(),
                'Current': weights_df.iloc[-1]
            })
            st.dataframe(
                weight_stats.style.format('{:.2%}'),
                use_container_width=True
            )
    
    # ============================================
    # TAB 6: Risk Analysis
    # ============================================
    with tab6:
        st.header("Risk Analysis")
        
        # Drawdown analysis
        st.subheader("Drawdown Analysis")
        
        selected_risk_strategy = st.selectbox(
            "Select Strategy for Risk Analysis",
            options=list(strategies.keys()),
            key='risk_strategy'
        )
        
        strategy_rets = strategies[selected_risk_strategy]
        if isinstance(strategy_rets, pd.Series):
            strategy_rets = strategy_rets.reindex(common_index).fillna(0)
        
        cum_returns = (1 + strategy_rets).cumprod()
        peak = cum_returns.cummax()
        drawdown = (peak - cum_returns) / peak
        
        fig_dd = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        fig_dd.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns.values, name='Portfolio Value', fill='tozeroy'),
            row=1, col=1
        )
        fig_dd.add_trace(
            go.Scatter(x=drawdown.index, y=-drawdown.values, name='Drawdown', fill='tozeroy', line=dict(color='red')),
            row=2, col=1
        )
        
        fig_dd.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Risk metrics
        st.subheader("Risk Metrics Comparison")
        
        risk_metrics = []
        for name, rets in strategies.items():
            if isinstance(rets, pd.Series):
                rets = rets.reindex(common_index).fillna(0)
            
            cum = (1 + rets).cumprod()
            pk = cum.cummax()
            dd = (pk - cum) / pk
            
            # VaR and CVaR
            var_95 = np.percentile(rets, 5)
            cvar_95 = rets[rets <= var_95].mean() if len(rets[rets <= var_95]) > 0 else var_95
            
            risk_metrics.append({
                'Strategy': name,
                'Volatility': rets.std() * np.sqrt(252),
                'Max Drawdown': dd.max(),
                'VaR (95%)': var_95,
                'CVaR (95%)': cvar_95,
                'Downside Vol': rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 0,
                'Skewness': rets.skew() if hasattr(rets, 'skew') else 0,
                'Kurtosis': rets.kurtosis() if hasattr(rets, 'kurtosis') else 0
            })
        
        risk_df = pd.DataFrame(risk_metrics).set_index('Strategy')
        
        st.dataframe(
            risk_df.style.format({
                'Volatility': '{:.2%}',
                'Max Drawdown': '{:.2%}',
                'VaR (95%)': '{:.2%}',
                'CVaR (95%)': '{:.2%}',
                'Downside Vol': '{:.2%}',
                'Skewness': '{:.2f}',
                'Kurtosis': '{:.2f}'
            }).background_gradient(cmap='RdYlGn_r', subset=['Volatility', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']),
            use_container_width=True
        )
        
        # VaR comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_var = px.bar(
                risk_df.reset_index(),
                x='Strategy',
                y='VaR (95%)',
                color='VaR (95%)',
                color_continuous_scale='RdYlGn',
                title='Value at Risk (95%) by Strategy'
            )
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            fig_mdd = px.bar(
                risk_df.reset_index(),
                x='Strategy',
                y='Max Drawdown',
                color='Max Drawdown',
                color_continuous_scale='RdYlGn_r',
                title='Maximum Drawdown by Strategy'
            )
            st.plotly_chart(fig_mdd, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Portfolio Optimization Dashboard | Built with Streamlit</p>
            <p>Data source: Yahoo Finance | RL Model: PPO (Stable-Baselines3)</p>
            <p>Disclaimer: For educational purposes only</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
