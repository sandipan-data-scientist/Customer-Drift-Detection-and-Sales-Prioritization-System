# Customer Drift Detection — Business Account Risk Monitoring

## What This Project Does

This project analyses historical transaction data from a UK-based online retailer and flags
business accounts whose purchasing behaviour has dropped in a statistically significant way.
Flagged accounts are grouped into four Prioritization Tiers so the Sales team can act on the
right customers in the right order.

## Project Layout

project/
    data.csv                  raw transaction data (place here before running)
    requirements.txt
    Dockerfile
    docker-compose.yml
    README.md
    notebooks/
        anomaly_detection.ipynb   full end-to-end notebook
    api/
        main.py                   FastAPI application
    streamlit_app/
        app.py                    Streamlit dashboard
    outputs/                      created by the notebook at runtime
        isolation_forest.pkl
        scaler.pkl
        tiered_customers.csv
        *.png                     all EDA and results plots


## How to Run Locally (Development)

Step 1: Install dependencies

    python -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install -r requirements.txt

Step 2: Place data.csv in the project root.

Step 3: Run the notebook end-to-end

    jupyter notebook notebooks/anomaly_detection.ipynb

    Run all cells from top to bottom. The notebook will:
      - clean and engineer features from the raw CSV
      - run statistical tests and print results
      - train and save the Isolation Forest model
      - produce all plots in outputs/
      - save tiered_customers.csv

Step 4: Start the API

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    Test it: open http://localhost:8000/docs in your browser.
    This opens the auto-generated Swagger UI where every endpoint can be tested interactively.

Step 5: Start the Streamlit dashboard

    streamlit run streamlit_app/app.py

    Open http://localhost:8501


## How to Run with Docker

Prerequisite: Docker and Docker Compose installed.

Step 1: Run the notebook to produce the outputs/ folder (Step 3 above).
        The container expects pre-built model files — it does not re-train at startup.

Step 2: Build and start both services

    docker-compose up --build

Step 3: Access the services

    API         http://localhost:8000
    API docs    http://localhost:8000/docs
    Dashboard   http://localhost:8501


## API Reference

GET  /health
    Returns: { "status": "ok", "records_loaded": N }

GET  /summary
    Returns executive summary: total customers, at-risk count, tier breakdown, Tier 1 value.

GET  /customers?tier=Tier+1&limit=50&offset=0
    Returns paginated customer list. Optional tier filter.

GET  /customers/{customer_id}
    Returns full feature profile for a single customer.

GET  /tiers
    Returns count of accounts in each tier.

POST /predict
    Body (JSON):
    {
        "days_since_last_transaction": 45,
        "total_lifetime_spend": 2000.0,
        "avg_monthly_spend": 300.0,
        "spend_last_30_days": 50.0,
        "drop_in_spend_vs_30_day_avg": 0.83,
        "recency_score": 0.35,
        "tx_frequency_per_month": 1.5,
        "avg_basket_size": 150.0,
        "active_months": 6
    }

    Returns:
    {
        "anomaly_flag": -1,
        "anomaly_score": -0.0812,
        "is_at_risk": true,
        "recommended_tier": "Tier 1: High Risk / High Value"
    }


## Prioritization Tiers Explained

Tier 1 — High Risk / High Value
    Your most valuable accounts showing the strongest anomaly signal.
    Act within 48 hours. Assign a senior account manager. Propose a custom retention offer.

Tier 2 — Low Risk / High Value
    Valuable accounts with early-stage warning signals.
    Schedule a check-in call this week. Offer a loyalty reward or early renewal discount.

Tier 3 — High Risk / Low Value
    Strong risk signal but lower revenue impact.
    Enroll in a 3-touch automated re-engagement email sequence over 30 days.

Tier 4 — Low Risk / Low Value
    Mild signal, low value. Standard monitoring.
    Include in the next scheduled marketing campaign. Reassess in 60 days.


## Statistical Validation

The anomaly signal is validated by two independent tests before the model runs:

t-test
    Compares average monthly spend between customers with high spend drops and those with low drops.
    If p < 0.05, the drop is statistically significant and not attributable to normal monthly variance.

Chi-Square test
    Tests whether the proportion of customers with severe drops (> 50% below their average)
    is independent of country segment (UK vs International).
    If p < 0.05, country segment is a meaningful risk factor.


## Retraining

To retrain with a newer dataset:
    1. Replace data.csv in the project root.
    2. Re-run the notebook from top to bottom.
    3. Rebuild the Docker image: docker-compose up --build


## Requirements

Python 3.11 or later
Docker 24 or later (for containerised deployment)
8 GB RAM recommended for the full dataset (541K rows)