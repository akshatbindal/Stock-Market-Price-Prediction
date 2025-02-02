# LSTM model trainer for stock market prediction

## Steps to run locally.

### 1. Clone the Repository
```bash
git clone https://github.com/akshatbindal/Stock-Market-Price-Prediction.git
cd Stock-Market-Price-Prediction
```

### 2. Install Dependencies
Make sure you have Python installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
Start the Streamlit app to interact with the model:
```bash
streamlit run app.py
```

### 4. Access the App
Open your web browser and go to the URL provided by Streamlit.

## Inputs for the Streamlit App 📝
The Streamlit app requires the following inputs:
- **Stock Ticker**: The ticker symbol of the stock you want to predict.
- **Date Range**: The start and end dates for the prediction period.
- **Model Parameters**: Optional parameters to fine-tune the LSTM model, such as the number of epochs, batch size, etc.

By providing these inputs, the app will preprocess the data, train the model if necessary, and display the predicted stock prices along with visualizations.

## Overview of LSTM Model 🧠
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is well-suited for time series prediction tasks. Unlike traditional RNNs, LSTMs can capture long-term dependencies and patterns in sequential data, making them ideal for stock market price prediction. The model consists of memory cells that can maintain information over long periods, gates that control the flow of information, and an output layer that generates predictions based on the processed data. By training on historical stock prices, the LSTM model learns to predict future prices with a certain degree of accuracy.
