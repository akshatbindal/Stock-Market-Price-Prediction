import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import fetch_data, scale_data
from feature_engineer import calculate_technical_indicators, create_dataset
from model import build_lstm_model
from utils import get_callbacks, evaluate_model
import numpy as np
import pandas as pd

def main():
    st.title("Stock Prediction with LSTM")

    # Sidebar Inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Ticker Symbol", value="^NSEI")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2014-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-01"))
    window_size = st.sidebar.slider("Window Size", 10, 50, 20)

    # Advanced Parameters
    st.sidebar.header("Advanced Parameters")
    neurons = st.sidebar.slider("Neurons", 10, 200, 100)
    layers = st.sidebar.slider("Layers", 1, 5, 1)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2)
    optimizer = st.sidebar.selectbox("Optimizer", ["adam", "sgd", "rmsprop"])
    loss = st.sidebar.selectbox("Loss Function", ["mean_squared_error", "mean_absolute_error"])

    # Initialize session state
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "selected_features" not in st.session_state:
        st.session_state["selected_features"] = []

    # Fetch data
    if st.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            data = fetch_data(ticker, start_date, end_date)
            data = calculate_technical_indicators(data)
            st.session_state["data"] = data  # Save data in session state
            st.success("Data fetched successfully!")

    # Ensure data is loaded
    if st.session_state["data"] is not None:
        data = st.session_state["data"]

        # Display feature selection
        st.header("Select Features")
        available_features = data.columns.tolist()
        st.session_state["selected_features"] = st.multiselect(
            "Available Features",
            options=available_features,
            default=["Close"],
        )

        # Start Training button
        if st.button("Start Training"):
            selected_features = st.session_state["selected_features"]
            if not selected_features:
                st.error("Please select at least one feature.")
                return

            # Filter data to selected features
            data = data[selected_features].dropna()

            # Scale and prepare data
            with st.spinner("Preparing data..."):
                scaled_data, scaler = scale_data(data, selected_features)
                X, y = create_dataset(scaled_data, window_size)

                # Split data into train and test sets
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

            # Train model
            with st.spinner("Training model..."):
                params = {'neurons': neurons, 'layers': layers, 'dropout': dropout, 'optimizer': optimizer, 'loss': loss}
                model = build_lstm_model((X_train.shape[1], X_train.shape[2]), params)
                callbacks = get_callbacks()
                model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)

            # Generate predictions
            with st.spinner("Generating predictions..."):
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                train_pred_rescaled = scaler.inverse_transform(
                    np.concatenate((train_pred, np.zeros((train_pred.shape[0], data.shape[1] - 1))), axis=1)
                )[:, 0]
                test_pred_rescaled = scaler.inverse_transform(
                    np.concatenate((test_pred, np.zeros((test_pred.shape[0], data.shape[1] - 1))), axis=1)
                )[:, 0]

                y_train_rescaled = scaler.inverse_transform(
                    np.concatenate((y_train.reshape(-1, 1), np.zeros((y_train.shape[0], data.shape[1] - 1))), axis=1)
                )[:, 0]
                y_test_rescaled = scaler.inverse_transform(
                    np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], data.shape[1] - 1))), axis=1)
                )[:, 0]

            # Evaluate model
            _, _, _, test_mape = evaluate_model(y_test_rescaled, test_pred_rescaled)
            st.success(f"Model trained successfully! Test MAPE: {test_mape:.2f}%")

            # Plot Results
            st.subheader("Predictions vs Actual")
            fig, ax = plt.subplots(2, 1, figsize=(12, 8))
            ax[0].plot(y_train_rescaled, label="Actual (Train)")
            ax[0].plot(train_pred_rescaled, label="Predicted (Train)")
            ax[0].legend()
            ax[0].set_title("Train Data")

            ax[1].plot(y_test_rescaled, label="Actual (Test)")
            ax[1].plot(test_pred_rescaled, label="Predicted (Test)")
            ax[1].legend()
            ax[1].set_title("Test Data")

            st.pyplot(fig)

if __name__ == "__main__":
    main()
