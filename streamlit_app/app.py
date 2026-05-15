"""
Streamlit dashboard for the Transaction Anomaly Detection project.
Connects to the FastAPI backend for all data.
Runs on: streamlit run streamlit_app/app.py
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

API_BASE = "http://localhost:8000"   # change to container URL in production

st.set_page_config(page_title="Transaction Anomaly Detector", layout="wide")
st.title("Transaction Anomaly Detection Dashboard")
st.caption("Business Account Risk Monitoring and Sales Prioritization")

# ---- sidebar ---- #
st.sidebar.header("Filters")
tier_options = [
    "All",
    "Tier 1: High Risk / High Value",
    "Tier 2: Low Risk / High Value",
    "Tier 3: High Risk / Low Value",
    "Tier 4: Low Risk / Low Value",
]
selected_tier = st.sidebar.selectbox("Filter by Tier", tier_options)

# ---- summary cards ---- #
try:
    summary = requests.get(f"{API_BASE}/summary", timeout=5).json()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{summary['total_customers']:,}")
    col2.metric("At-Risk Accounts", f"{summary['at_risk_count']}")
    col3.metric("At-Risk Percentage", f"{summary['at_risk_pct']}%")
    col4.metric("Tier 1 Combined Value", f"£{summary['tier1_combined_value']:,.0f}")
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

st.markdown("---")

# ---- tier counts bar chart ---- #
st.subheader("Account Distribution by Tier")
tier_data = requests.get(f"{API_BASE}/tiers", timeout=5).json()
tier_df   = pd.DataFrame(list(tier_data.items()), columns=["Tier", "Count"])
tier_colors = {
    "Tier 1: High Risk / High Value" : "#e74c3c",
    "Tier 2: Low Risk / High Value"  : "#f39c12",
    "Tier 3: High Risk / Low Value"  : "#8e44ad",
    "Tier 4: Low Risk / Low Value"   : "#2980b9",
    "No Action"                      : "#bdc3c7",
}
colors = [tier_colors.get(t, "#95a5a6") for t in tier_df["Tier"]]
fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(tier_df["Tier"], tier_df["Count"], color=colors)
ax.set_ylabel("Number of Accounts")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ---- customer table ---- #
st.subheader("At-Risk Customer List")

tier_filter = None if selected_tier == "All" else selected_tier
params      = {"limit": 200, "offset": 0}
if tier_filter:
    params["tier"] = tier_filter

cust_resp = requests.get(f"{API_BASE}/customers", params=params, timeout=10).json()
cust_df   = pd.DataFrame(cust_resp["data"])

if not cust_df.empty and "priority_tier" in cust_df.columns:
    at_risk = cust_df[cust_df["priority_tier"] != "No Action"]
    display_cols = [
        "CustomerID", "total_lifetime_spend", "days_since_last_transaction",
        "drop_in_spend_vs_30_day_avg", "anomaly_score", "priority_tier"
    ]
    available = [c for c in display_cols if c in at_risk.columns]
    st.dataframe(at_risk[available].sort_values("anomaly_score"), use_container_width=True)
    st.caption(f"Showing {len(at_risk)} at-risk accounts")
else:
    st.info("No at-risk accounts found for the selected filter.")

st.markdown("---")

# ---- scatter plot ---- #
st.subheader("Risk Landscape: Anomaly Score vs Lifetime Value")
if not cust_df.empty and "anomaly_score" in cust_df.columns:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for tier, color in tier_colors.items():
        subset = cust_df[cust_df.get("priority_tier", pd.Series()) == tier] if "priority_tier" in cust_df.columns else pd.DataFrame()
        if not subset.empty:
            ax2.scatter(
                subset["anomaly_score"],
                subset["total_lifetime_spend"],
                c=color, label=tier, alpha=0.7, s=30
            )
    ax2.set_xlabel("Anomaly Score")
    ax2.set_ylabel("Lifetime Spend (£)")
    ax2.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")

# ---- predict a new customer ---- #
st.subheader("Score a New Customer Account")
with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    days_since   = c1.number_input("Days Since Last Transaction", 0, 365, 45)
    lifetime     = c1.number_input("Total Lifetime Spend (£)", 0.0, 1000000.0, 2000.0)
    avg_monthly  = c2.number_input("Avg Monthly Spend (£)", 0.0, 100000.0, 300.0)
    last30       = c2.number_input("Spend Last 30 Days (£)", 0.0, 100000.0, 50.0)
    drop_ratio   = c3.number_input("Drop vs 30-Day Avg (0=none, 1=total drop)", 0.0, 1.0, 0.7)
    recency_sc   = c3.number_input("Recency Score (0-1)", 0.0, 1.0, 0.3)
    freq         = c1.number_input("Transactions Per Month", 0.0, 50.0, 1.5)
    basket       = c2.number_input("Avg Basket Size (£)", 0.0, 5000.0, 150.0)
    active_mo    = c3.number_input("Active Months", 1, 24, 6)
    submitted    = st.form_submit_button("Run Prediction")

if submitted:
    payload = {
        "days_since_last_transaction" : days_since,
        "total_lifetime_spend"        : lifetime,
        "avg_monthly_spend"           : avg_monthly,
        "spend_last_30_days"          : last30,
        "drop_in_spend_vs_30_day_avg" : drop_ratio,
        "recency_score"               : recency_sc,
        "tx_frequency_per_month"      : freq,
        "avg_basket_size"             : basket,
        "active_months"               : active_mo,
    }
    result = requests.post(f"{API_BASE}/predict", json=payload, timeout=5).json()
    if result["is_at_risk"]:
        st.error(f"AT RISK: {result['recommended_tier']}  |  Score: {result['anomaly_score']:.4f}")
    else:
        st.success(f"Normal account  |  Score: {result['anomaly_score']:.4f}")