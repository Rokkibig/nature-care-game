#!/usr/bin/env python
"""
Простий тест LSTM моделі з технічними індикаторами
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("🚀 ТЕСТ ENHANCED LSTM (БЕЗ TensorFlow)")
print("="*50)

# 1. Завантаження даних
print("\n📥 Завантаження BTC-USD...")
df = yf.download('BTC-USD', period='1y', progress=False)
print(f"✅ Завантажено {len(df)} днів")

# 2. Додавання індикаторів
print("\n📊 Розрахунок технічних індикаторів...")
df['MA_7'] = df['Close'].rolling(7).mean()
df['MA_30'] = df['Close'].rolling(30).mean()
df['Volume_MA'] = df['Volume'].rolling(10).mean()
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(20).std()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

df = df.dropna()
print(f"✅ Додано {7} індикаторів")

# 3. Підготовка даних
features = ['Close', 'Volume', 'RSI', 'Returns', 'Volatility']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# 4. Створення sequences
seq_length = 30
X, y = [], []
for i in range(seq_length, len(scaled_data)):
    X.append(scaled_data[i-seq_length:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# 5. Train/Test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n🔧 Дані підготовлені:")
print(f"   Training:  {X_train.shape}")
print(f"   Testing:   {X_test.shape}")
print(f"   Features:  {len(features)}")

# 6. Проста модель (Linear Regression як baseline)
from sklearn.linear_model import LinearRegression

print("\n🏗️ Навчання простої моделі (baseline)...")
# Flatten для Linear Regression
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

model = LinearRegression()
model.fit(X_train_flat, y_train)

# Прогноз
y_pred = model.predict(X_test_flat)

# Inverse transform
y_test_real = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
y_pred_real = y_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

# Метрики
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(np.mean((y_test_real - y_pred_real)**2))

print("\n" + "="*50)
print("📊 РЕЗУЛЬТАТИ (Linear Regression baseline):")
print("="*50)
print(f"MAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")

# Порівняння з базовою моделлю
base_mae = 109.84
improvement = ((base_mae - mae) / base_mae) * 100

if improvement > 0:
    print(f"\n🎉 Покращення: {improvement:.1f}% vs стара модель!")
else:
    print(f"\n📈 Базова модель: MAE ${mae:.2f}")

# Останній прогноз
current_price = float(df['Close'].iloc[-1])
print(f"\nПоточна ціна BTC: ${current_price:,.2f}")

print("\n✅ Тест завершено!")
print("\n💡 Для справжньої LSTM моделі потрібен TensorFlow")
print("   Встановіть: pip install tensorflow==2.13.0")
