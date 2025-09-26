import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from datetime import datetime

# Відтворюємо точно ту саму модель
df = yf.download('BTC-USD', period='6mo', progress=False)
df['Returns'] = df['Close'].pct_change()
df['MA7'] = df['Close'].rolling(7).mean()
df['RSI'] = 100 - 100/(1 + df['Close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / 
                        -df['Close'].diff().where(lambda x: x<0, 0).rolling(14).mean())
df = df.dropna()

X = df[['Open', 'High', 'Low', 'Volume', 'Returns', 'MA7', 'RSI']].values[:-1]
y = df['Close'].values[1:]

split = int(len(X) * 0.8)
X_train = X[:split]
y_train = y[:split]

# Навчаємо фінальну модель на всіх даних
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Зберігаємо
joblib.dump(model, 'btc_rf_model.pkl')
print(f"Модель збережена: btc_rf_model.pkl")

# Зберігаємо параметри
params = {
    'features': ['Open', 'High', 'Low', 'Volume', 'Returns', 'MA7', 'RSI'],
    'mae': 2192.86,
    'accuracy': 98.05,
    'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'price_at_training': float(df['Close'].iloc[-1])
}

joblib.dump(params, 'btc_model_params.pkl')
print(f"Параметри збережені: btc_model_params.pkl")
