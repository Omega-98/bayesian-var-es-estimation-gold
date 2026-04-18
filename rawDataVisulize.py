import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data
file_path = 'Gold_Futures_Historical_Data.csv'
df = pd.read_csv(file_path)


df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)


df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))

# visualize
df_plot = df.dropna()

plt.figure(figsize=(12, 8))

# trend
plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['Price'], color='gold')
plt.title('Gold Price Trend')
plt.grid(True, alpha=0.3)

# distribution
plt.subplot(2, 1, 2)
sns.histplot(df_plot['Log_Return'], bins=100, kde=True, color='blue')
plt.title('Log Returns Distribution (Check for Heavy Tails)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Data is successfully converted! Log Returns first 5 lines:")
print(df_plot['Log_Return'].head())