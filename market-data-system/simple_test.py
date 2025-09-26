import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("🚀 Тест простої моделі (без TensorFlow)\n")

# 1. Завантаження даних
print("📥 Завантаження BTC-USD...")
df = yf.download('BTC-USD', period='3mo', progress=False)
print(f"✅ Завантажено {len(df)} днів")

# 2. Прості індикатори
df['MA_7'] = df['Close'].rolling(7).mean()
df['Returns'] = df['Close'].pct_change()
df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(7).mean()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

df = df.dropna()
print(f"✅ Додано індикатори\n")

# 3. Підготовка features
features = ['MA_7', 'Volume_Ratio', 'RSI', 'Returns']
X = df[features].values[:-1]  # всі крім останнього
y = df['Close'].values[1:]    # зміщено на 1

# Split 
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Проста модель (Linear Regression)
print("🏗️ Навчання простої моделі...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Прогноз
y_pred = model.predict(X_test)

# Метрики
mae = np.mean(np.abs(y_test - y_pred))
print(f"✅ MAE: ${mae:.2f}")

# Останній прогноз
last_features = df[features].iloc[-1].values.reshape(1, -1)
next_price = model.predict(last_features)[0]
current_price = df['Close'].iloc[-1]

print(f"\n📊 Результати:")
print(f"Поточна ціна: ${current_price:.2f}")
print(f"Прогноз:      ${next_price:.2f}")
print(f"Зміна:        {((next_price-current_price)/current_price*100):+.1f}%")

print("\n✅ Тест завершено! yfinance та sklearn працюють.")
