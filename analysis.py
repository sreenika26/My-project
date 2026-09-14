import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

df = pd.read_csv("sales_data.csv", parse_dates=["date"])

report_lines = []

def log(line=""):
    print(line)
    report_lines.append(line)

log("=== DATA CLEANING ===")
log(f"Raw rows: {len(df)}")

before_dupes = len(df)
df = df.drop_duplicates()
log(f"Removed {before_dupes - len(df)} exact duplicate rows")

df["store"] = df["store"].str.strip().str.title()

missing_prices = df["unit_price"].isna().sum()
df["unit_price"] = df.groupby("category")["unit_price"].transform(
    lambda s: s.fillna(s.median())
)
log(f"Filled {missing_prices} missing unit_price values with category median")

price_z = df.groupby("category")["unit_price"].transform(
    lambda s: (s - s.mean()) / s.std()
)
outliers = df[price_z.abs() > 3]
log(f"Flagged {len(outliers)} price outliers (|z| > 3 within category), removing them")
df = df[price_z.abs() <= 3].copy()

df["revenue"] = df["unit_price"] * df["units_sold"]
log(f"Clean rows: {len(df)}")

log("\n=== OVERALL KPIs ===")
total_revenue = df["revenue"].sum()
total_units = df["units_sold"].sum()
avg_order_value = df["revenue"].mean()
log(f"Total revenue: ${total_revenue:,.2f}")
log(f"Total units sold: {total_units:,}")
log(f"Average transaction value: ${avg_order_value:,.2f}")

log("\n=== REVENUE BY STORE ===")
by_store = df.groupby("store")["revenue"].sum().sort_values(ascending=False)
log(by_store.to_string())

log("\n=== REVENUE BY CATEGORY ===")
by_category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
log(by_category.to_string())

monthly = df.set_index("date").resample("ME")["revenue"].sum()
monthly_growth = monthly.pct_change() * 100

log("\n=== MONTH-OVER-MONTH GROWTH (%) ===")
log(monthly_growth.round(2).to_string())

rolling_30d = df.set_index("date").resample("D")["revenue"].sum().rolling(30).mean()

pivot = pd.pivot_table(
    df, values="revenue", index="store", columns="category", aggfunc="sum", fill_value=0
)
log("\n=== STORE x CATEGORY REVENUE PIVOT ===")
log(pivot.to_string())

best_store_category = pivot.stack().idxmax()
log(f"\nBest store/category combo: {best_store_category[0]} - {best_store_category[1]} "
    f"(${pivot.stack().max():,.2f})")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

by_category.plot(kind="bar", ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Revenue by Category")
axes[0, 0].set_ylabel("Revenue ($)")
axes[0, 0].tick_params(axis="x", rotation=30)

by_store.plot(kind="bar", ax=axes[0, 1], color="darkorange")
axes[0, 1].set_title("Revenue by Store")
axes[0, 1].set_ylabel("Revenue ($)")
axes[0, 1].tick_params(axis="x", rotation=0)

monthly.plot(ax=axes[1, 0], marker="o", color="green")
axes[1, 0].set_title("Monthly Revenue Trend")
axes[1, 0].set_ylabel("Revenue ($)")

rolling_30d.plot(ax=axes[1, 1], color="purple")
axes[1, 1].set_title("30-Day Rolling Average Daily Revenue")
axes[1, 1].set_ylabel("Revenue ($)")

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150)
log("\nSaved chart panel -> dashboard.png")

with open("analysis_report.txt", "w") as f:
    f.write("\n".join(report_lines))