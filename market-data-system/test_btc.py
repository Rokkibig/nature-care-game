#!/usr/bin/env python
import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.executable}\n")

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    print(f"🚀 Тест BTC аналізу - {datetime.now().strftime('%H:%M')}\n")
    
    # Завантаження даних
    df = yf.download('BTC-USD', period='1mo', progress=False)
    
    # Прості індикатори
    df['MA7'] = df['Close'].rolling(7).mean()
    df['Change'] = df['Close'].pct_change() * 100
    
    # Статистика
    current = df['Close'].iloc[-1]
    ma7 = df['MA7'].iloc[-1]
    change_24h = df['Change'].iloc[-1]
    
    print("📊 Bitcoin (BTC-USD)")
    print(f"Ціна:      ${current:,.2f}")
    print(f"MA(7):     ${ma7:,.2f}")
    print(f"Зміна 24h: {change_24h:+.2f}%")
    print("\n✅ Все працює!")
    
except ImportError as e:
    print(f"❌ Помилка імпорту: {e}")
    print("\nВстановіть пакети через:")
    print("/Users/abbastudio_/miniconda3/envs/lstm/bin/pip install yfinance numpy pandas")
