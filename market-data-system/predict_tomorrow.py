import yfinance as yf
import numpy as np
import joblib

# Завантажуємо модель
model = joblib.load('btc_rf_model.pkl')

# Отримуємо свіжі дані
df = yf.download('BTC-USD', period='1mo', progress=False)
df['Returns'] = df['Close'].pct_change()
df['MA7'] = df['Close'].rolling(7).mean()
df['RSI'] = 100 - 100/(1 + df['Close'].diff().where(lambda x: x>0, 0).rolling(14).mean() / 
                        -df['Close'].diff().where(lambda x: x<0, 0).rolling(14).mean())
df = df.dropna()

# Останні дані для прогнозу
last_data = df[['Open', 'High', 'Low', 'Volume', 'Returns', 'MA7', 'RSI']].iloc[-1].values.reshape(1, -1)

# Прогноз
predicted_price = model.predict(last_data)[0]
current_price = float(df['Close'].iloc[-1])
change = (predicted_price - current_price) / current_price * 100

print("="*50)
print("ПРОГНОЗ НА ЗАВТРА:")
print("="*50)
print(f"Поточна ціна:  ${current_price:,.2f}")
print(f"Прогноз:       ${predicted_price:,.2f}")
print(f"Очікувана зміна: {change:+.2f}%")
print("="*50)
