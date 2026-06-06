import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =====================
# LOAD DATA
# =====================

df = pd.read_csv(
    "data/stockmaster_preprocessed.csv"
)

# =====================
# TARGET
# =====================

data = df["Target"].values.reshape(-1,1)

# =====================
# SCALING
# =====================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

# =====================
# SEQUENCE
# =====================

X = []
y = []

window = 60

for i in range(window, len(scaled_data)):
    X.append(
        scaled_data[i-window:i]
    )

    y.append(
        scaled_data[i]
    )

X = np.array(X)
y = np.array(y)

# =====================
# SPLIT
# =====================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =====================
# MODEL
# =====================

model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(X_train.shape[1],1)
    )
)

model.add(
    LSTM(50)
)

model.add(
    Dense(25)
)

model.add(
    Dense(1)
)

# =====================
# COMPILE
# =====================

model.compile(
    optimizer="adam",
    loss="mse"
)

# =====================
# TRAIN
# =====================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test,y_test)
)

# =====================
# SAVE MODEL
# =====================

model.save(
    "models/lstm_model.h5"
)

print("\nLSTM model berhasil disimpan")