"""
Efficient Frontier Visualizer with CAL and International Market Support

This Streamlit web app lets users select securities (US and international) and calculates
the Efficient Frontier along with a Capital Allocation Line (CAL) that is tangent to the
frontier at the maximum Sharpe ratio portfolio. The app now displays company names along
with ticker symbols and includes robust error handling.

Key improvements in this version:
1. Improved extraction of price data: If "Adj Close" is not available, the app will use "Close".
2. Excellent inline documentation and comments.
3. Persistent ticker storage using a JSON file.
"""

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
    if os.path.exists(TICKER_FILE):
        with open(TICKER_FILE, 'r') as f:
            st.session_state.tickers = json.load(f)
    else:
        st.session_state.tickers = []

def get_ticker_info(ticker):
    """
    Returns a string with the ticker and its company name.
    If company info is unavailable, it returns 'Unknown Company'.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        company_name = ticker_obj.info.get('longName', 'Unknown Company')
    except Exception:
        company_name = "Unknown Company"
    return f"{ticker} - {company_name}"

def update_ticker_file():
    """Updates the persistent storage (JSON file) with the current ticker list."""
    with open(TICKER_FILE, 'w') as f:
        json.dump(st.session_state.tickers, f)

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculates portfolio performance metrics.
    Returns annualized volatility, return, and Sharpe ratio.
    """
    returns = np.sum(mean_returns * weights) * 252  # 252 trading days/year
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return std_dev, returns, sharpe_ratio

def efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.01, num_portfolios=10000):
    """
    Simulates portfolios to build the Efficient Frontier.
    Returns a results array, record of weights, and calculates the Capital Allocation Line (CAL).
    """
    results = np.zeros((4, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        std_dev, returns, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = std_dev
        results[1, i] = returns
        results[2, i] = sharpe_ratio
        weights_record.append(weights)
    
    # Identify the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_weights = weights_record[max_sharpe_idx]
    max_sharpe_std_dev, max_sharpe_return, _ = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
    
    # Calculate the Capital Allocation Line (CAL)
    cal_x = np.linspace(0, 2 * max_sharpe_std_dev, 100)
    cal_y = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_std_dev * cal_x
    # Prepare hover text to display the capital allocation percentages
    cal_hover_text = [f"Allocation: {dict(zip(st.session_state.tickers, max_sharpe_weights.round(2)))}" for _ in cal_x]
    
    return results, weights_record, max_sharpe_weights, cal_x, cal_y, cal_hover_text

# ------------------ Streamlit Interface ------------------

st.title("Global Efficient Frontier Visualizer with CAL and International Market Support")
st.write("Select securities (US and international), add or remove tickers, and visualize the Efficient Frontier along with the Capital Allocation Line (CAL).")

# Display current securities with company names
st.write("### Current securities in portfolio:")
st.write([get_ticker_info(ticker) for ticker in st.session_state.tickers])

# Add a ticker (with help text for international formats)
ticker_to_add = st.text_input("Enter a stock ticker to add (e.g., 2330.TW for TSMC, 1050.SR for TASI stock):", "")
if st.button("Add Ticker") and ticker_to_add:
    ticker_to_add = ticker_to_add.upper()
    if ticker_to_add not in st.session_state.tickers:
        st.session_state.tickers.append(ticker_to_add)
        st.success(f"{get_ticker_info(ticker_to_add)} added to the portfolio.")
        update_ticker_file()
    else:
        st.warning(f"{get_ticker_info(ticker_to_add)} is already in the portfolio.")

# Remove a ticker using a dropdown that displays company names
ticker_to_remove = st.selectbox(
    "Select a ticker to remove",
    options=[get_ticker_info(ticker) for ticker in st.session_state.tickers]
)
if st.button("Remove Selected Ticker") and ticker_to_remove:
    # Extract the ticker symbol from the formatted string "TICKER - Company Name"
    ticker_symbol = ticker_to_remove.split(" - ")[0]
    st.session_state.tickers.remove(ticker_symbol)
    st.success(f"{ticker_to_remove} removed from the portfolio.")
    update_ticker_file()

if st.session_state.tickers:
    # Date inputs for the historical data range
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    
    if st.button("Calculate Efficient Frontier"):
        # Download data from Yahoo Finance for the selected tickers
        data = yf.download(st.session_state.tickers, start=start_date, end=end_date)
        
        # ----- FIX: Handle price data extraction for one or multiple tickers -----
        # Sometimes, the downloaded DataFrame does not contain 'Adj Close'. 
        # In that case, we try to use 'Close' as a fallback.
        if isinstance(data.columns, pd.MultiIndex):
            # For multiple tickers, check for 'Adj Close' in level 1; if missing, try 'Close'
            if 'Adj Close' in data.columns.get_level_values(1):
                data = data.xs('Adj Close', axis=1, level=1)
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', axis=1, level=1)
            else:
                st.error("The downloaded data does not contain 'Adj Close' or 'Close'. Please check the ticker symbols.")
                st.stop()
        else:
            # For a single ticker, check if 'Adj Close' is present; if not, use 'Close'
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                st.error("The downloaded data does not contain 'Adj Close' or 'Close'. Please check the ticker symbol.")
                st.stop()
        # ------------------------------------------------------------------------
        
        # Fill missing values to align the data
        if data.isnull().values.any():
            st.warning("Some data is missing for the selected tickers. Filling missing values forward.")
            data = data.fillna(method='ffill').dropna()
        
        # Verify that sufficient data is available
        if data.empty or (isinstance(data, pd.DataFrame) and len(data.columns) < len(st.session_state.tickers)):
            st.error("Data could not be retrieved for some tickers. Please check the ticker symbols and try again.")
        else:
            daily_returns = data.pct_change()
            mean_returns = daily_returns.mean()
            cov_matrix = daily_returns.cov()
            
            # Calculate the Efficient Frontier and CAL
            results, weights_record, max_sharpe_weights, cal_x, cal_y, cal_hover_text = efficient_frontier(mean_returns, cov_matrix)
            
            # Identify portfolios with the maximum Sharpe ratio and minimum volatility
            max_sharpe_idx = np.argmax(results[2])
            max_sharpe_ratio = results[:, max_sharpe_idx]
            min_vol_idx = np.argmin(results[0])
            min_vol_ratio = results[:, min_vol_idx]
            
            # Build the interactive Plotly figure
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=results[0, :],
                    y=results[1, :],
                    mode='markers',
                    marker=dict(color=results[2, :], colorscale='Viridis', colorbar=dict(title='Sharpe Ratio')),
                    text=[f"Return: {r:.2f}, Volatility: {v:.2f}, Sharpe Ratio: {s:.2f}"
                          for r, v, s in zip(results[1, :], results[0, :], results[2, :])],
                    hoverinfo='text',
                    name="Portfolio Points"
                )
            )
            # Highlight the Max Sharpe Ratio portfolio
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
            # Highlight the Minimum Volatility portfolio
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
            # Add the Capital Allocation Line (CAL)
            fig.add_trace(
                go.Scatter(
                    x=cal_x,
                    y=cal_y,
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dash'),
                    name="Capital Allocation Line (CAL)",
                    text=cal_hover_text,
                    hoverinfo='text'
                )
            )
            fig.update_layout(
                title="Efficient Frontier with Capital Allocation Line",
                xaxis_title="Volatility",
                yaxis_title="Return",
                legend=dict(x=0.8, y=1.2)
            )
            st.plotly_chart(fig)
            
            # Display portfolio details
            st.write("### Max Sharpe Ratio Portfolio")
            st.write("Annualized Return:", max_sharpe_ratio[1])
            st.write("Annualized Volatility:", max_sharpe_ratio[0])
            st.write("Capital Allocation:", dict(zip(st.session_state.tickers, max_sharpe_weights.round(2))))
            
            st.write("### Min Volatility Portfolio")
            st.write("Annualized Return:", min_vol_ratio[1])
            st.write("Annualized Volatility:", min_vol_ratio[0])
