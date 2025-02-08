"""
Efficient Frontier Visualizer with CAL, International Market Support, and High Growth Screening

Screen High Growth Stocks: Click the “Screen High Growth Stocks” button to automatically 
fill your portfolio with tickers that have demonstrated strong revenue and profit growth 
(from the candidate list).

This Streamlit web app lets users select securities (US and international) and calculates
the Efficient Frontier along with a Capital Allocation Line (CAL) tangent to the frontier at 
the maximum Sharpe ratio portfolio. The app displays company names along with ticker symbols,
stores tickers persistently, and now includes a fundamental screening option to automatically 
select high-growth stocks based on revenue growth and profit margins over the last 5 years.

Key Improvements in This Version:
1. Robust price data extraction using auto_adjust=True so that adjusted price data is returned in "Close".
2. Added a "Screen High Growth Stocks" button to automatically select stocks that show strong revenue 
   and net profit (profit margin) growth.
3. Comprehensive inline documentation for clarity and maintainability.
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
    If company info is unavailable, returns 'Unknown Company'.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        company_name = ticker_obj.info.get('longName', 'Unknown Company')
    except Exception:
        company_name = "Unknown Company"
    return f"{ticker} - {company_name}"

def update_ticker_file():
    """Updates the persistent JSON storage with the current ticker list."""
    with open(TICKER_FILE, 'w') as f:
        json.dump(st.session_state.tickers, f)

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculates portfolio performance metrics:
      - Annualized return
      - Annualized volatility (std dev)
      - Sharpe ratio
    """
    returns = np.sum(mean_returns * weights) * 252  # Assume 252 trading days per year
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return std_dev, returns, sharpe_ratio

def efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.01, num_portfolios=10000):
    """
    Simulates many portfolios to build the Efficient Frontier.
    Also calculates the Capital Allocation Line (CAL) tangent at the maximum Sharpe ratio portfolio.
    Returns:
      - results: array of volatility, return, and Sharpe ratio for each portfolio
      - weights_record: list of portfolio weights
      - max_sharpe_weights: weights of the max Sharpe ratio portfolio
      - cal_x, cal_y: points for the CAL line
      - cal_hover_text: text to display on hover over the CAL
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
    
    # Find the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_weights = weights_record[max_sharpe_idx]
    max_sharpe_std_dev, max_sharpe_return, _ = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
    
    # Calculate the Capital Allocation Line (CAL)
    cal_x = np.linspace(0, 2 * max_sharpe_std_dev, 100)
    cal_y = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_std_dev * cal_x
    cal_hover_text = [f"Allocation: {dict(zip(st.session_state.tickers, max_sharpe_weights.round(2)))}" for _ in cal_x]
    
    return results, weights_record, max_sharpe_weights, cal_x, cal_y, cal_hover_text

def filter_high_growth_stocks(tickers, min_revenue_growth=0.05, min_profit_margin=0.1):
    """
    Screens a list of candidate tickers and returns only those that meet the criteria for
    high revenue growth and profit margins.
    
    Parameters:
      - tickers: list of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
      - min_revenue_growth: minimum revenue growth (as a decimal, e.g., 0.05 for 5%)
      - min_profit_margin: minimum profit margin (as a decimal, e.g., 0.1 for 10%)
      
    Note: The function uses fundamental data from Yahoo Finance. Some tickers may not have
          the required fields; those are skipped.
    """
    selected = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rev_growth = info.get("revenueGrowth")
            profit_margin = info.get("profitMargins")
            # If both values are available and exceed thresholds, select the ticker.
            if rev_growth is not None and profit_margin is not None:
                if rev_growth > min_revenue_growth and profit_margin > min_profit_margin:
                    selected.append(ticker)
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")
    return selected

# ------------------ Streamlit Interface ------------------

st.title("Global Efficient Frontier Visualizer with CAL and High Growth Screening")
st.write("Select securities (US and international), add or remove tickers, and visualize the Efficient Frontier with the Capital Allocation Line (CAL).")
st.write("Optionally, use the fundamental screen to automatically add high-growth stocks (with revenue and net profit growth over the last 5 years).")

# Display current securities with company names
st.write("### Current securities in portfolio:")
st.write([get_ticker_info(ticker) for ticker in st.session_state.tickers])

# Create columns for removal buttons
col1, col2 = st.columns(2)
with col1:
    # Remove a single ticker using a dropdown
    ticker_to_remove = st.selectbox(
        "Select a ticker to remove",
        options=[get_ticker_info(ticker) for ticker in st.session_state.tickers]
    )
    if st.button("Remove Selected Ticker"):
        ticker_symbol = ticker_to_remove.split(" - ")[0]
        st.session_state.tickers.remove(ticker_symbol)
        st.success(f"{ticker_to_remove} removed from the portfolio.")
        update_ticker_file()

with col2:
    # Button to clear all tickers from the portfolio
    if st.button("Clear All Tickers"):
        st.session_state.tickers = []
        update_ticker_file()
        st.success("All tickers removed. Your portfolio is now empty.")

# Add a ticker manually
ticker_to_add = st.text_input("Enter a stock ticker to add (e.g., 2330.TW for TSMC, 1050.SR for TASI stock):", "")
if st.button("Add Ticker") and ticker_to_add:
    ticker_to_add = ticker_to_add.upper()
    if ticker_to_add not in st.session_state.tickers:
        st.session_state.tickers.append(ticker_to_add)
        st.success(f"{get_ticker_info(ticker_to_add)} added to the portfolio.")
        update_ticker_file()
    else:
        st.warning(f"{get_ticker_info(ticker_to_add)} is already in the portfolio.")

# New: Button to screen and add high growth stocks automatically
if st.button("Screen High Growth Stocks"):
    # Define a candidate list (you can update this list as needed)
    candidate_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA", "ADBE", "CRM", "ORCL", "INTC"]
    filtered = filter_high_growth_stocks(candidate_tickers, min_revenue_growth=0.05, min_profit_margin=0.1)
    if filtered:
        st.session_state.tickers = list(set(st.session_state.tickers + filtered))
        update_ticker_file()
        st.success("High growth stocks added: " + ", ".join(filtered))
    else:
        st.warning("No candidate tickers met the high growth criteria.")

# Only run calculations if there are tickers in the portfolio
if st.session_state.tickers:
    # Date input for historical price data
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    
    if st.button("Calculate Efficient Frontier"):
        # Download price data with auto_adjust=True to return adjusted "Close" prices
        data = yf.download(st.session_state.tickers, start=start_date, end=end_date, auto_adjust=True)
        
        # Robust price data extraction: handle MultiIndex columns if multiple tickers were downloaded
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                data = data["Close"]
            else:
                st.error("The downloaded data does not contain 'Close'. Please check the ticker symbols.")
                st.stop()
        else:
            if "Close" in data.columns:
                data = data["Close"]
            else:
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                else:
                    st.error("The downloaded data does not contain 'Close'. Please check the ticker symbol.")
                    st.stop()
        
        # Fill missing data if necessary
        if data.isnull().values.any():
            st.warning("Some data is missing. Filling missing values forward.")
            data = data.fillna(method='ffill').dropna()
        
        # Verify sufficient data was retrieved
        if data.empty or (isinstance(data, pd.DataFrame) and len(data.columns) < len(st.session_state.tickers)):
            st.error("Data could not be retrieved for some tickers. Please check the ticker symbols and try again.")
        else:
            daily_returns = data.pct_change()
            mean_returns = daily_returns.mean()
            cov_matrix = daily_returns.cov()
            
            # Calculate the Efficient Frontier and CAL
            results, weights_record, max_sharpe_weights, cal_x, cal_y, cal_hover_text = efficient_frontier(mean_returns, cov_matrix)
            
            # Identify portfolios with maximum Sharpe ratio and minimum volatility
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
