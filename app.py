#
# Author: Ahmed Albaqawi
# This code is my first AI generated code to evaluate building a code that can run on dynamic environments
# like Streamlit Community Cloud or Replit or Deta or AWS
#
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import json
import os

# File to store persistent ticker memory
TICKER_FILE = 'tickers.json'

# Initialize session state for tickers if not already set
if 'tickers' not in st.session_state:
    # Load saved tickers or initialize empty list
    if os.path.exists(TICKER_FILE):
        with open(TICKER_FILE, 'r') as f:
            st.session_state.tickers = json.load(f)
    else:
        st.session_state.tickers = []

# Function to calculate portfolio performance metrics
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252  # 252 trading days per year
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std_dev, returns

# Function to calculate Efficient Frontier
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        std_dev, returns = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = std_dev
        results[1,i] = returns
        results[2,i] = (returns - 0.01) / std_dev  # Assuming risk-free rate of 1%
        weights_record.append(weights)
    return results, weights_record

# Update persistent memory function
def update_ticker_file():
    with open(TICKER_FILE, 'w') as f:
        json.dump(st.session_state.tickers, f)

# Streamlit Interface
st.title("Efficient Frontier Visualizer with Persistent Memory")
st.write("Select securities for the portfolio, add or remove tickers, and visualize the Efficient Frontier dynamically.")

# Display current securities and options to add/remove
st.write("### Current securities in portfolio:")
st.write(st.session_state.tickers)

# Add ticker
ticker_to_add = st.text_input("Enter a stock ticker to add:", "")
if st.button("Add Ticker") and ticker_to_add:
    ticker_to_add = ticker_to_add.upper()
    if ticker_to_add not in st.session_state.tickers:
        st.session_state.tickers.append(ticker_to_add)
        st.success(f"{ticker_to_add} added to the portfolio.")
        update_ticker_file()
    else:
        st.warning(f"{ticker_to_add} is already in the portfolio.")

# Remove ticker
ticker_to_remove = st.selectbox("Select a ticker to remove", st.session_state.tickers)
if st.button("Remove Selected Ticker") and ticker_to_remove:
    st.session_state.tickers.remove(ticker_to_remove)
    st.success(f"{ticker_to_remove} removed from the portfolio.")
    update_ticker_file()

# Proceed with calculations only if there are tickers
if st.session_state.tickers:
    # Date inputs for historical data range
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    
    if st.button("Calculate Efficient Frontier"):
        # Fetch data from Yahoo Finance
        data = yf.download(st.session_state.tickers, start=start_date, end=end_date)['Adj Close']
        daily_returns = data.pct_change()
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        
        # Efficient Frontier Calculation
        results, weights_record = efficient_frontier(mean_returns, cov_matrix)
        
        # Maximum Sharpe Ratio and Minimum Volatility Portfolios
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_ratio = results[:, max_sharpe_idx]
        min_vol_idx = np.argmin(results[0])
        min_vol_ratio = results[:, min_vol_idx]
        
        # Create Plotly scatter plot for Efficient Frontier
        fig = go.Figure()

        # Add all portfolio points
        fig.add_trace(
            go.Scatter(
                x=results[0,:],
                y=results[1,:],
                mode='markers',
                marker=dict(color=results[2,:], colorscale='Viridis', colorbar=dict(title='Sharpe Ratio')),
                text=[f"Return: {r:.2f}, Volatility: {v:.2f}, Sharpe Ratio: {s:.2f}" for r, v, s in zip(results[1,:], results[0,:], results[2,:])],
                hoverinfo='text',
                name="Portfolio Points"
            )
        )

        # Highlight Max Sharpe Ratio and Min Volatility portfolios
        fig.add_trace(
            go.Scatter(
                x=[max_sharpe_ratio[0]],
                y=[max_sharpe_ratio[1]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='star'),
                name='Max Sharpe Ratio',
                hovertext=f"Return: {max_sharpe_ratio[1]:.2f}, Volatility: {max_sharpe_ratio[0]:.2f}"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[min_vol_ratio[0]],
                y=[min_vol_ratio[1]],
                mode='markers',
                marker=dict(color='blue', size=12, symbol='star'),
                name='Min Volatility',
                hovertext=f"Return: {min_vol_ratio[1]:.2f}, Volatility: {min_vol_ratio[0]:.2f}"
            )
        )

        # Layout settings
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Volatility",
            yaxis_title="Return",
            legend=dict(x=0.8, y=1.2)
        )

        # Display the Plotly chart in Streamlit
        st.plotly_chart(fig)

        # Display portfolio details
        st.write("### Max Sharpe Ratio Portfolio")
        st.write("Annualized Return:", max_sharpe_ratio[1])
        st.write("Annualized Volatility:", max_sharpe_ratio[0])
        
        st.write("### Min Volatility Portfolio")
        st.write("Annualized Return:", min_vol_ratio[1])
        st.write("Annualized Volatility:", min_vol_ratio[0])
