"""
BrightBank EDA - Snowflake-powered exploration
Pushes all aggregations to Snowflake; only small result sets are fetched locally.
"""

import polars as pl
import snowflake.connector
from plotnine import *

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

conn = snowflake.connector.connect(connection_name="workbench")
cursor = conn.cursor()
cursor.execute("USE WAREHOUSE CI_WH")

TABLE = "LENDING_CLUB.BRIGHTBANK.SYNTHETIC_CUSTOMERS"

# ---------------------------------------------------------------------------
# 1. Segment distribution
# ---------------------------------------------------------------------------

cursor.execute(f"""
    SELECT SEGMENT, COUNT(*) AS COUNT
    FROM {TABLE}
    GROUP BY SEGMENT
    ORDER BY COUNT DESC
""")
seg_df = pl.DataFrame(cursor.fetchall(), schema=["SEGMENT", "COUNT"], orient="row")
print("=== Segment Distribution ===")
print(seg_df)

# ---------------------------------------------------------------------------
# 2. Key metrics by segment
# ---------------------------------------------------------------------------

cursor.execute(f"""
    SELECT
        SEGMENT,
        ROUND(AVG(ANNUAL_INCOME), 0)       AS AVG_INCOME,
        ROUND(MEDIAN(ANNUAL_INCOME), 0)    AS MEDIAN_INCOME,
        MIN(ANNUAL_INCOME)                 AS MIN_INCOME,
        MAX(ANNUAL_INCOME)                 AS MAX_INCOME,
        ROUND(AVG(CREDIT_SCORE), 1)        AS AVG_CREDIT_SCORE,
        ROUND(AVG(CHURN_RISK_SCORE), 1)    AS AVG_CHURN_RISK,
        ROUND(AVG(ACCOUNT_BALANCE), 0)     AS AVG_BALANCE,
        ROUND(AVG(NUM_PRODUCTS), 2)        AS AVG_PRODUCTS
    FROM {TABLE}
    GROUP BY SEGMENT
    ORDER BY AVG_INCOME DESC
""")
cols = [desc[0] for desc in cursor.description]
stats_df = pl.DataFrame(cursor.fetchall(), schema=cols, orient="row")
print("\n=== Key Metrics by Segment ===")
print(stats_df)

# ---------------------------------------------------------------------------
# 3. Product holding rates and churn by region
# ---------------------------------------------------------------------------

cursor.execute(f"""
    SELECT
        REGION,
        COUNT(*)                                                             AS CUSTOMERS,
        ROUND(AVG(CASE WHEN HAS_MORTGAGE    THEN 1.0 ELSE 0 END) * 100, 1) AS PCT_MORTGAGE,
        ROUND(AVG(CASE WHEN HAS_INVESTMENT  THEN 1.0 ELSE 0 END) * 100, 1) AS PCT_INVESTMENT,
        ROUND(AVG(CASE WHEN HAS_CREDIT_CARD THEN 1.0 ELSE 0 END) * 100, 1) AS PCT_CREDIT_CARD,
        ROUND(AVG(CHURN_RISK_SCORE), 1)                                      AS AVG_CHURN_RISK,
        ROUND(AVG(ANNUAL_INCOME), 0)                                         AS AVG_INCOME
    FROM {TABLE}
    GROUP BY REGION
    ORDER BY CUSTOMERS DESC
""")
cols = [desc[0] for desc in cursor.description]
region_df = pl.DataFrame(cursor.fetchall(), schema=cols, orient="row")
print("\n=== Product Holdings & Churn by Region ===")
print(region_df)

# ---------------------------------------------------------------------------
# 4. Visualisations (plotnine)
# ---------------------------------------------------------------------------

# 4a. Segment distribution
p_segment = (
    ggplot(seg_df.sort("COUNT"), aes(x="reorder(SEGMENT, COUNT)", y="COUNT"))
    + geom_col(fill="steelblue")
    + coord_flip()
    + labs(title="Customer Segment Distribution", x="Segment", y="Count")
)
p_segment

# 4b. Average income by segment
p_income = (
    ggplot(stats_df, aes(x="reorder(SEGMENT, AVG_INCOME)", y="AVG_INCOME"))
    + geom_col(fill="steelblue")
    + coord_flip()
    + labs(title="Average Annual Income by Segment", x="Segment", y="Avg Income ($)")
)
p_income

# 4c. Average churn risk by region
p_churn = (
    ggplot(region_df, aes(x="reorder(REGION, AVG_CHURN_RISK)", y="AVG_CHURN_RISK"))
    + geom_col(fill="tomato")
    + coord_flip()
    + labs(title="Avg Churn Risk Score by Region", x="Region", y="Avg Churn Risk Score")
)
p_churn

# 4d. Product holding rates by region (long format for grouped bars)
product_long = region_df.select(
    "REGION", "PCT_MORTGAGE", "PCT_INVESTMENT", "PCT_CREDIT_CARD"
).unpivot(
    index="REGION",
    variable_name="PRODUCT",
    value_name="PCT",
)
p_products = (
    ggplot(product_long, aes(x="REGION", y="PCT", fill="PRODUCT"))
    + geom_col(position="dodge")
    + labs(
        title="Product Holding Rates by Region (%)",
        x="Region",
        y="% of Customers",
        fill="Product",
    )
)
p_products

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cursor.close()
conn.close()
