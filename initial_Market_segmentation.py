import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import datetime

# Load dataset
df = pd.read_csv("data.csv")

# ----------------- Basic Exploration -----------------
print("Data shape:", df.shape)
print("Column names:", df.columns.tolist())
print(df.head())
print(df.info())

# ----------------- Data Cleaning -----------------
# Convert arrival_date to datetime
df['arrival_date'] = pd.to_datetime(df['arrival_date'], dayfirst=True)

# Create a price range column
df['price_range'] = df['max_price'] - df['min_price']

# ----------------- Aggregations & Insights -----------------
# Average modal price by state
state_avg_price = df.groupby('state')['modal_price'].mean().sort_values(ascending=False).head(5)
print("Top 5 states by average modal price:\n", state_avg_price)

# High volatility commodities
high_volatility = df.groupby('commodity')['price_range'].mean().sort_values(ascending=False).head(5)
print("Commodities with highest average price range:\n", high_volatility)

# ----------------- Visualization -----------------
# Barplot for top 5 states by modal price
plt.figure(figsize=(10, 6))
sns.barplot(x=state_avg_price.values, y=state_avg_price.index, palette="viridis")
plt.title('Top 5 States by Average Modal Price')
plt.xlabel('Average Modal Price')
plt.ylabel('State')
plt.tight_layout()
plt.show()

# ----------------- Clustering (District + Commodity) -----------------
# Optional: apply KMeans clustering based on modal price and price range
clustering_data = df[['modal_price', 'price_range']].dropna()
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(clustering_data)

# ----------------- Time-Series Exploration -----------------
# Example: Price trend for a specific commodity over time
banana_data = df[df['commodity'].str.contains('Banana', case=False)]
banana_ts = banana_data.groupby('arrival_date')['modal_price'].mean()

plt.figure(figsize=(12, 6))
banana_ts.plot()
plt.title('Modal Price Trend of Banana Over Time')
plt.xlabel('Date')
plt.ylabel('Modal Price')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------- Export Cleaned Data -----------------
df.to_csv("cleaned_agri_data.csv", index=False)
