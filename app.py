## This is the fix for:
## 1. Missing data handling dates that lead to NAN for holidays or mix match in trading days
## for international markets.
## 2. Also Error Checks for data retreivals 
###############################################################
###############################################################
## These changes are completed to add the following
## 1. Interactive Capital Allocation Line (CAL)
## 2. International Market Tickers. 
## Users can enter international tickers in the format used by Yahoo Finance 
## (e.g., 2330.TW for TSMC on the Taiwan Stock Exchange or 1050.SR for Saudi Arabian companies in TASI).
## All FINA data is fetched using Yahoo! Finance library - yfinance
##
##
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
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    returns = np.sum(mean_returns * weights) * 252  # 252 trading days per year
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return std_dev, returns, sharpe_ratio

# Function to calculate Efficient Frontier and Capital Allocation Line
def efficient_frontier(mean_returns, cov_matrix, risk_free_rate=0.01, num_portfolios=10000):
    results = np.zeros((4, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        std_dev, returns, sharpe_ratio = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0,i] = std_dev
        results[1,i] = returns
        results[2,i] = sharpe_ratio
        weights_record.append(weights)
    
    # Find max Sharpe Ratio portfolio and calculate CAL
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_weights = weights_record[max_sharpe_idx]
    max_sharpe_std_dev, max_sharpe_return, _ = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
    cal_x = np.linspace(0, 2 * max_sharpe_std_dev, 100)
    cal_y = risk_free_rate + (max_sharpe_return - risk_free_rate) / max_sharpe_std_dev * cal_x
    cal_hover_text = [f"Allocation: {dict(zip(st.session_state.tickers, max_sharpe_weights.round(2)))}" for _ in cal_x]
    
    return results, weights_record, max_sharpe_weights, cal_x, cal_y, cal_hover_text

# Update persistent memory function
def update_ticker_file():
    with open(TICKER_FILE, 'w') as f:
        json.dump(st.session_state.tickers, f)

# Streamlit Interface
st.title("Global Efficient Frontier Visualizer with CAL and International Market Support")
st.write("Select securities from any market, add/remove tickers, and visualize the Efficient Frontier with the CAL.")

# Display current securities and options to add/remove
st.write("### Current securities in portfolio:")
st.write(st.session_state.tickers)

# Add ticker
ticker_to_add = st.text_input("Enter a stock ticker to add (e.g., 2330.TW for TSMC, 1050.SR for TASI stock):", "")
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
        # Fetch data from Yahoo Finance (support for international markets)
        data = yf.download(st.session_state.tickers, start=start_date, end=end_date)['Adj Close']
        
        # Check for missing data
        if data.isnull().values.any():
            st.warning("Some data is missing for the selected tickers. Filling missing values forward.")
            data = data.fillna(method='ffill').dropna()  # Fill missing data and remove remaining NA rows
        
        # Ensure we have sufficient data
        if data.empty or len(data.columns) < len(st.session_state.tickers):
            st.error("Data could not be retrieved for some tickers. Please check the ticker symbols and try again.")
        else:
            daily_returns = data.pct_change()
            mean_returns = daily_returns.mean()
            cov_matrix = daily_returns.cov()
            
            # Calculate Efficient Frontier and CAL
            results, weights_record, max_sharpe_weights, cal_x, cal_y, cal_hover_text = efficient_frontier(mean_returns, cov_matrix)
            
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

            # Add CAL
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

            # Layout settings
            fig.update_layout(
                title="Efficient Frontier with Capital Allocation Line",
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
            st.write("Capital Allocation:", dict(zip(st.session_state.tickers, max_sharpe_weights.round(2))))
            
            st.write("### Min Volatility Portfolio")
            st.write("Annualized Return:", min_vol_ratio[1])
            st.write("Annualized Volatility:", min_vol_ratio[0])
