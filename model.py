from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, params):
    """Build an LSTM model."""
    model = Sequential()
    model.add(LSTM(params['neurons'], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(params['dropout']))
    for _ in range(params['layers'] - 1):
        model.add(LSTM(params['neurons'], return_sequences=True))
        model.add(Dropout(params['dropout']))
    model.add(LSTM(params['neurons'], return_sequences=False))
    model.add(Dropout(params['dropout']))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    return model
