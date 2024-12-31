import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fetch_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    return data

def scale_data(data, features):
    """Scale the data using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    return scaled_data, scaler
