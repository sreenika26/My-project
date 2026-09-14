import numpy as np
import pandas as pd

np.random.seed(42)

stores = ["North", "South", "East", "West", "Central"]
categories = {
    "Electronics": (150, 900),
    "Groceries": (5, 60),
    "Clothing": (15, 120),
    "Home & Garden": (20, 300),
    "Toys": (10, 80),
}

dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
rows = []
for date in dates:
    n_transactions = np.random.poisson(25)
    for _ in range(n_transactions):
        store = np.random.choice(stores)
        category = np.random.choice(list(categories.keys()))
        low, high = categories[category]
        base_price = np.random.uniform(low, high)
        seasonal_boost = 1.4 if date.month == 12 and category in ["Electronics", "Toys"] else 1.0
        price = round(base_price * seasonal_boost, 2)
        units = np.random.randint(1, 6)
        rows.append([date, store, category, price, units])

df = pd.DataFrame(rows, columns=["date", "store", "category", "unit_price", "units_sold"])

missing_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
df.loc[missing_idx, "unit_price"] = np.nan

dup_rows = df.sample(frac=0.01, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

bad_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[bad_idx, "unit_price"] = df.loc[bad_idx, "unit_price"] * 25

case_idx = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
df.loc[case_idx, "store"] = df.loc[case_idx, "store"].str.lower()

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.to_csv("sales_data.csv", index=False)
print(f"Generated {len(df)} rows -> sales_data.csv")