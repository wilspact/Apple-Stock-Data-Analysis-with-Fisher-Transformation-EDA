import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import skew,kurtosis
df=pd.read_csv("C:\\Users\\Lenovo\\Desktop\\dataset\\project\\kurtosis and skewness\\sandp500.csv")
print(df.info())
symbol='AAPL'
stock=df[df['Name']==symbol].sort_values('date')
stock['Return']=stock["close"].pct_change().dropna()
skw=skew(stock['Return'].dropna())
krt=kurtosis(stock['Return'].dropna(),fisher=False)
print(f"Skewness for {symbol}: {skw:.4f}")
print(f"Kurtosis for {symbol}: {krt:.4f}")
plt.figure(figsize=(12,5))
sns.histplot(stock['Return'].dropna(),bins=50,kde=True)
plt.title(f'{symbol} Daily Returns Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()
stock['LogReturn']=np.log1p(stock['Return'])
skw_log=skew(stock['LogReturn'].dropna())
krt_log = kurtosis(stock['LogReturn'].dropna(), fisher=False)
print(f"Log-Return Skewness: {skw_log:.4f}")
print(f"Log-Return Kurtosis: {krt_log:.4f}")
plt.figure(figsize=(12,5))
sns.histplot(stock['LogReturn'].dropna(), bins=50, kde=True)
plt.title(f'{symbol} Log-Transformed Return Distribution')
plt.xlabel('Log Daily Return')
plt.ylabel('Frequency')
plt.show()
