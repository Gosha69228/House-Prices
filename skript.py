import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_dat = pd.read_csv('train.csv')
test_dat = pd.read_csv('test.csv')
df = pd.concat([train_dat, test_dat], ignore_index=True, sort=False)

# уникальные значения
uniq_counts = df.nunique()
pd.set_option('display.max_rows', None)
print(uniq_counts)
print()
print("//////////////////////////////////////////////////")
print()
# просмотр пропусков
D = df.isnull().sum()
for column_name, count in D.items():
    if count != 0:
        print(f"{column_name} {count}")

plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Visualization of Missing Values')
# plt.show()

sale = df["GarageYrBlt"].value_counts()
print(sale)

print(max(df['LotFrontage']))
print(min(df['LotFrontage']))

correlation = df['LotFrontage'].corr(df['LotArea'])
print(correlation)

print(min(df['LotFrontage']))
print(max(df['LotFrontage']))