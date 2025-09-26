import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = yf.download('BTC-USD', period='6mo', progress=False)

# Features engineering
df['Returns'] = df['Close'].pct_change()
df['MA7'] = df['Close'].rolling(7).mean()
df['RSI'] = 100 - 100/(1 + df['Close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / 
                        -df['Close'].diff().where(lambda x: x<0, 0).rolling(14).mean())
df = df.dropna()

X = df[['Open', 'High', 'Low', 'Volume', 'Returns', 'MA7', 'RSI']].values[:-1]
y = df['Close'].values[1:]

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"🎯 Random Forest MAE: ${mae:.2f}")
print(f"📊 Точність: {100 - (mae/y_test.mean()*100):.2f}%")
print(f"💰 Поточна ціна: ${df['Close'].iloc[-1]:,.2f}")
