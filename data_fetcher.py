import pandas as pd
import numpy as np


df = pd.read_csv('AAPL_2020_2025_daily.csv', skiprows=3)
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df[['Close', 'Volume']].dropna()

df['Return'] = df['Close'].pct_change().copy()

def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Close'])

df['Vol_MA20'] = df['Volume'].rolling(20, min_periods=1).mean().copy()
df['Volume_Bin'] = np.where(
    df['Volume'] > df['Vol_MA20'] * 1.2, 2,
    np.where(df['Volume'] < df['Vol_MA20'] * 0.8, 0, 1)
)

df['SMA3'] = df['Close'].rolling(3).mean()
df['Price_Trend'] = np.where(
    df['SMA3'].diff() > 0, 1,  
    np.where(df['SMA3'].diff() < 0, -1, 0) 
)

df = df.dropna()
df.to_csv('AAPL_data.csv')