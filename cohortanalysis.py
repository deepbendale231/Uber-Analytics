# cohort_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect
from urllib.parse import quote_plus

# ---------- EDIT ONLY THIS BLOCK ----------
db_user = "root"
db_password = "Poojadeep@231"   # <- paste your password here exactly
db_host = "localhost"
db_port = 3306
db_name = "uber_analytics"
table_name = "ncr_ride_bookings"
# -----------------------------------------

password_enc = quote_plus(db_password)
conn_str = f"mysql+mysqlconnector://{db_user}:{password_enc}@{db_host}:{db_port}/{db_name}"
print("Using connection string (password hidden) ->", conn_str.replace(password_enc, "*****"))

# create engine and test connection
engine = create_engine(conn_str, pool_pre_ping=True)
ins = inspect(engine)

try:
    tables = ins.get_table_names()
    print("Connected. Tables in DB:", tables)
    if table_name not in tables:
        raise SystemExit(f"Table '{table_name}' not found in database '{db_name}'. Stop and check the table name in Workbench.")
except Exception as e:
    print("ERROR connecting or listing tables:", type(e).__name__, e)
    raise

# Load completed rides (only necessary columns)
query = f"""
SELECT `Customer ID`, `Date`
FROM `{table_name}`
WHERE `Booking Status` = 'Completed';
"""

try:
    df_cohort = pd.read_sql(query, engine)
except Exception as e:
    print("ERROR reading SQL:", type(e).__name__, e)
    raise

if df_cohort.empty:
    raise SystemExit("No completed rides were returned by the query. Check the table & Booking Status values.")

# Convert to datetime and compute months
df_cohort['Date'] = pd.to_datetime(df_cohort['Date'], errors='coerce')
df_cohort = df_cohort.dropna(subset=['Date','Customer ID'])
df_cohort['SignupMonth'] = df_cohort.groupby('Customer ID')['Date'].transform('min').dt.to_period('M')
df_cohort['RideMonth'] = df_cohort['Date'].dt.to_period('M')

# Cohort index: 1 = signup month, 2 = next month, etc.
df_cohort['CohortIndex'] = (
    (df_cohort['RideMonth'].dt.year - df_cohort['SignupMonth'].dt.year) * 12 +
    (df_cohort['RideMonth'].dt.month - df_cohort['SignupMonth'].dt.month) + 1
)

# Aggregate unique customers per cohort month & cohort index
cohort_data = (
    df_cohort.groupby(['SignupMonth', 'CohortIndex'])['Customer ID']
    .nunique()
    .reset_index(name='n_customers')
)

cohort_pivot = cohort_data.pivot_table(
    index='SignupMonth',
    columns='CohortIndex',
    values='n_customers'
).fillna(0).astype(int)

# retention percent (divide each row by cohort size)
cohort_size = cohort_pivot.iloc[:, 0]
retention = cohort_pivot.div(cohort_size, axis=0)

# Print small checks
print("\nCohort pivot (counts) — first 8 rows:")
print(cohort_pivot.head(8))
print("\nRetention (fractions) — first 8 rows:")
print(retention.head(8))

# Save retention table to CSV (optional)
cohort_pivot.to_csv("cohort_counts.csv")
retention.to_csv("cohort_retention_fraction.csv")

# Plot heatmap (formatted as percent)
plt.figure(figsize=(12, 6))
sns.heatmap(retention, annot=True, fmt=".0%", cmap="Blues", cbar=True)
plt.title("Cohort Analysis — Customer Retention Over Time")
plt.ylabel("Signup Month")
plt.xlabel("Months Since Signup (1 = signup month)")
plt.tight_layout()
plt.show()
