import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
file_path = 'Gold_Futures_Historical_Data.csv'
df = pd.read_csv(file_path)

# 2. 核心修复：强制转换 Price 列为数字
# 首先确保它是字符串，然后去掉逗号，最后转成 float
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)

# 3. 日期转换与排序
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# 4. 计算对数收益率 (现在 Price 已经是数字了，不会再报错)
df['Log_Return'] = np.log(df['Price'] / df['Price'].shift(1))

# 5. 可视化
df_plot = df.dropna()

plt.figure(figsize=(12, 8))

# 价格走势
plt.subplot(2, 1, 1)
plt.plot(df['Date'], df['Price'], color='gold')
plt.title('Gold Price Trend')
plt.grid(True, alpha=0.3)

# 收益率分布
plt.subplot(2, 1, 2)
sns.histplot(df_plot['Log_Return'], bins=100, kde=True, color='blue')
plt.title('Log Returns Distribution (Check for Heavy Tails)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("数据转换成功！Log Returns 前五行：")
print(df_plot['Log_Return'].head())