#!/usr/bin/env python
"""
Enhanced LSTM Model with Technical Indicators
Виправлена версія для нової yfinance
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 ENHANCED LSTM MODEL WITH TECHNICAL INDICATORS")
print("="*60)

# 1. Завантаження даних
print("\n📥 Завантаження BTC-USD...")
df = yf.download('BTC-USD', period='2y', progress=False)
print(f"✅ Завантажено {len(df)} днів")

# Виправлення для нової версії yfinance
# Якщо колонки мають MultiIndex, розпакуємо їх
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 2. Технічні індикатори
print("\n📊 Розрахунок технічних індикаторів...")

# Moving Averages
df['MA_7'] = df['Close'].rolling(7).mean()
df['MA_14'] = df['Close'].rolling(14).mean()
df['MA_30'] = df['Close'].rolling(30).mean()

# Price ratios - виправлено
df['Price_MA7_Ratio'] = df['Close'].values / df['MA_7'].values
df['Price_MA30_Ratio'] = df['Close'].values / df['MA_30'].values

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

# MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Volume indicators
df['Volume_MA'] = df['Volume'].rolling(10).mean()
df['Volume_Ratio'] = df['Volume'].values / df['Volume_MA'].values

# Volatility
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(20).std()

# Price action
df['High_Low_Ratio'] = df['High'].values / df['Low'].values
df['Close_Open_Ratio'] = df['Close'].values / df['Open'].values

df = df.dropna()
print(f"✅ Додано 13 технічних індикаторів")

# 3. Вибір features
features = [
    'Close', 'Volume', 'RSI', 'MACD', 'Signal',
    'Price_MA7_Ratio', 'Price_MA30_Ratio', 
    'Volume_Ratio', 'Returns', 'Volatility',
    'High_Low_Ratio', 'Close_Open_Ratio'
]

print(f"\n📈 Використовуємо {len(features)} features")

# 4. Підготовка даних
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Створення sequences
seq_length = 60
X, y = [], []
for i in range(seq_length, len(scaled_data)):
    X.append(scaled_data[i-seq_length:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Train/Test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n🔧 Дані підготовлені:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Testing:  {X_test.shape[0]} samples")

try:
    import tensorflow as tf
    print("\n✅ TensorFlow знайдено!")
    
    # Створення моделі
    print("🏗️ Створення LSTM моделі...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, 
                            input_shape=(seq_length, len(features))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print(f"✅ Модель створена: {model.count_params():,} параметрів")
    
    # Навчання
    print("\n🎯 Навчання (20 епох)...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Оцінка
    print("\n📈 Оцінка моделі...")
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_test_real = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    y_pred_real = y_pred.flatten() * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(np.mean((y_test_real - y_pred_real)**2))
    
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТИ LSTM:")
    print("="*60)
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    
    if mae < 109.84:
        improvement = ((109.84 - mae) / 109.84) * 100
        print(f"\n🎉 ПОКРАЩЕННЯ: {improvement:.1f}% краще!")
    
except ImportError:
    print("\n⚠️ TensorFlow не встановлено")
    print("🔄 Використовуємо Random Forest...")
    
    from sklearn.ensemble import RandomForestRegressor
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_flat, y_train)
    
    y_pred = model.predict(X_test_flat)
    y_test_real = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    y_pred_real = y_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(np.mean((y_test_real - y_pred_real)**2))
    
    print(f"\n📊 Random Forest результати:")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")

# Поточна ціна
current_price = float(df['Close'].iloc[-1])
print(f"\nПоточна ціна BTC: ${current_price:,.2f}")

print("\n✅ Готово!")
