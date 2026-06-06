import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# ======================
# LOAD DATA
# ======================

df = pd.read_csv(
    "data/stockmaster_preprocessed.csv"
)

# ======================
# FEATURES
# ======================

X = df[
[
    'Close_Price',
    'Basic EPS',
    'DER',
    'ROA',
    'ROE',
    'MA5',
    'MA20',
    'Volatility20'
]
]

y = df["Target"]

# ======================
# SPLIT DATA
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================
# MODEL
# ======================

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ======================
# PREDICTION
# ======================

pred = model.predict(X_test)

# ======================
# EVALUATION
# ======================

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = mean_squared_error(
    y_test,
    pred
) ** 0.5

r2 = r2_score(
    y_test,
    pred
)

print("\n===== XGBOOST RESULT =====")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# ======================
# SAVE MODEL
# ======================

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

print("\nModel berhasil disimpan")