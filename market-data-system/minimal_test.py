#!/usr/bin/env python3
"""
Мінімальний тест Enhanced LSTM - працює за 1-2 хвилини
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("🚀 Швидкий тест Enhanced LSTM\n")

# 1. Дані
print("📥 Завантаження BTC-USD...")
df = yf.download('BTC-USD', period='6mo', progress=False)
print(f"✅ Завантажено {len(df)} днів\n")

# 2. Прості індикатори
print("📊 Додавання індикаторів...")
df['MA_7'] = df['Close'].rolling(7).mean()
df['Volume_MA'] = df['Volume'].rolling(7).mean()
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(14).std()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

df = df.dropna()
print(f"✅ Додано 5 індикаторів\n")

# 3. Підготовка даних
features = ['Close', 'Volume', 'RSI', 'Returns', 'Volatility']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# Sequences
seq_len = 20
X, y = [], []
for i in range(seq_len, len(scaled)):
    X.append(scaled[i-seq_len:i])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"🔧 Дані готові: {X_train.shape}\n")

# 4. Модель
print("🏗️ Створення моделі...")
try:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(seq_len, len(features))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print(f"✅ Модель: {model.count_params()} параметрів\n")
    
    # 5. Навчання
    print("🎯 Навчання (5 епох)...")
    model.fit(X_train, y_train, 
             epochs=5, 
             batch_size=32, 
             validation_split=0.2,
             verbose=0)
    print("✅ Навчання завершено\n")
    
    # 6. Тест
    y_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform
    y_test_real = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    y_pred_real = y_pred.flatten() * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    
    # Результати
    mae = np.mean(np.abs(y_test_real - y_pred_real))
    rmse = np.sqrt(np.mean((y_test_real - y_pred_real)**2))
    
    print("="*40)
    print("📊 РЕЗУЛЬТАТИ:")
    print("="*40)
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    
    # Порівняння
    base_mae = 109.84
    if mae < base_mae:
        improvement = ((base_mae - mae) / base_mae) * 100
        print(f"\n🎉 Покращення: {improvement:.1f}% vs базова модель!")
    
    # Останній прогноз
    last_price = df['Close'].iloc[-1]
    last_pred = y_pred_real[-1]
    print(f"\nОстання ціна: ${last_price:.2f}")
    print(f"Прогноз:      ${last_pred:.2f}")
    
except ImportError:
    print("❌ Встановіть TensorFlow: pip install tensorflow")
except Exception as e:
    print(f"❌ Помилка: {e}")

print("\n✅ Готово!")
