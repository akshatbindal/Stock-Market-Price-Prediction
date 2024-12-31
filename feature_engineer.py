import ta
import numpy as np

def calculate_technical_indicators(data):
    """Add technical indicators to the dataset using the `ta` library."""
    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return data

def create_dataset(data, time_steps):
    """Create sequences of data for LSTM training."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])  # Assuming 'Close' is the target
    return np.array(X), np.array(y)
