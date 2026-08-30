import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    artifact = joblib.load(
        "champion_fraud_model.joblib"
    )

    return artifact


artifact = load_model()

model = artifact["pipeline"]

THRESHOLD = artifact["optimal_threshold"]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # Log transaction amount
    df["log_amount_usd"] = np.log1p(
        df["amount_usd"]
    )

    # Cumulative risk factor
    risk_columns = [
        "used_vpn",
        "ip_country_mismatch",
        "billing_shipping_mismatch",
        "is_foreign_transaction",
        "is_ai_generated_scam_attempt",
        "is_new_merchant"
    ]

    df["cumulitative_risk_factor"] = (
        df[risk_columns].sum(axis=1)
    )

    # Amount / balance ratio
    df["amount_to_balance_ratio"] = (
        df["amount_usd"] /
        (df["account_balance_usd"] + 1)
    )

    # Velocity / transaction gap
    df["velocity_to_gap_ratio"] = (
        df["velocity_score"] /
        (df["hours_since_last_txn"] + 0.1)
    )

    # High-risk merchant
    high_risk_merchants = [
        "Crypto Exchange",
        "Gift Cards",
        "Gaming"
    ]

    df["is_high_risk_merchant"] = (
        df["merchant_category"]
        .isin(high_risk_merchants)
    )

    # No authentication
    df["is_aunthenticated"] = (
        df["auth_method"] ==
        "No Authentication"
    )

    return df


# ============================================================
# TITLE
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    ### Fraud Risk Prediction

    Enter the details of a transaction below and the trained
    machine learning model will estimate its probability of fraud.
    """
)

st.divider()


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.subheader("Transaction Information")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    amount_usd = st.number_input(
        "Transaction Amount (USD)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        [
            "Crypto Exchange",
            "Gift Cards",
            "Gaming",
            "Electronics",
            "Fuel",
            "Groceries",
            "Healthcare",
            "Online Retail",
            "Restaurants",
            "Streaming",
            "Travel",
            "Utilities"
        ]
    )

    card_type = st.selectbox(
        "Card Type",
        [
            "Visa",
            "Mastercard",
            "Amex",
            "Discover",
            "RuPay"
        ]
    )

    auth_method = st.selectbox(
        "Authentication Method",
        [
            "3D Secure",
            "Biometric",
            "No Authentication",
            "OTP",
            "PIN"
        ]
    )

    channel = st.selectbox(
        "Transaction Channel",
        [
            "ATM",
            "Contactless",
            "In-App",
            "Online",
            "POS"
        ]
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    device_type = st.selectbox(
        "Device Type",
        [
            "ATM Machine",
            "Android Phone",
            "Mac",
            "POS Terminal",
            "Smart Watch",
            "Tablet",
            "Windows PC",
            "iPhone"
        ]
    )

    hours_since_last_txn = st.number_input(
        "Hours Since Last Transaction",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    txn_count_last_24h = st.number_input(
        "Transaction Count (Last 24h)",
        min_value=0,
        value=3,
        step=1
    )

    distance_from_home_km = st.number_input(
        "Distance From Home (km)",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    card_age_months = st.number_input(
        "Card Age (Months)",
        min_value=0,
        value=24,
        step=1
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        value=35,
        step=1
    )


# ------------------------------------------------------------
# COLUMN 3
# ------------------------------------------------------------

with col3:

    account_balance_usd = st.number_input(
        "Account Balance (USD)",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

    cvv_retry_count = st.number_input(
        "CVV Retry Count",
        min_value=0,
        value=0,
        step=1
    )

    velocity_score = st.number_input(
        "Velocity Score",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

    time_of_day_hour = st.slider(
        "Time of Day",
        0,
        23,
        14
    )

    day_of_week = st.slider(
        "Day of Week",
        0,
        6,
        2
    )

    merchant_risk_score = st.slider(
        "Merchant Risk Score",
        0.0,
        1.0,
        0.30,
        0.01
    )

    prior_disputes = st.number_input(
        "Prior Disputes",
        min_value=0,
        value=0,
        step=1
    )


# ============================================================
# RISK INDICATORS
# ============================================================

st.divider()

st.subheader("Risk Indicators")

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:

    is_foreign_transaction = st.checkbox(
        "Foreign Transaction"
    )

    is_new_merchant = st.checkbox(
        "New Merchant"
    )

with risk_col2:

    used_vpn = st.checkbox(
        "VPN Used"
    )

    ip_country_mismatch = st.checkbox(
        "IP Country Mismatch"
    )

with risk_col3:

    billing_shipping_mismatch = st.checkbox(
        "Billing / Shipping Mismatch"
    )

    is_ai_generated_scam_attempt = st.checkbox(
        "AI Generated Scam Attempt"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Fraud Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # Create one transaction
    transaction = pd.DataFrame([{

        "transaction_id": 1,

        "amount_usd":
            amount_usd,

        "merchant_category":
            merchant_category,

        "card_type":
            card_type,

        "auth_method":
            auth_method,

        "channel":
            channel,

        "device_type":
            device_type,

        "is_foreign_transaction":
            int(is_foreign_transaction),

        "hours_since_last_txn":
            hours_since_last_txn,

        "txn_count_last_24h":
            txn_count_last_24h,

        "distance_from_home_km":
            distance_from_home_km,

        "card_age_months":
            card_age_months,

        "customer_age":
            customer_age,

        "account_balance_usd":
            account_balance_usd,

        "is_new_merchant":
            int(is_new_merchant),

        "used_vpn":
            int(used_vpn),

        "ip_country_mismatch":
            int(ip_country_mismatch),

        "billing_shipping_mismatch":
            int(billing_shipping_mismatch),

        "cvv_retry_count":
            cvv_retry_count,

        "velocity_score":
            velocity_score,

        "time_of_day_hour":
            time_of_day_hour,

        "day_of_week":
            day_of_week,

        "is_ai_generated_scam_attempt":
            int(is_ai_generated_scam_attempt),

        "merchant_risk_score":
            merchant_risk_score,

        "prior_disputes":
            prior_disputes
    }])


    # Feature engineering
    transaction = create_features(
        transaction
    )


    # Remove columns that were removed during training
    transaction_model = transaction.drop(
        columns=[
            "transaction_id",
            "amount_usd",
            "account_balance_usd",
            "velocity_score"
        ],
        errors="ignore"
    )


    # Prediction probability
    probability = model.predict_proba(
        transaction_model
    )[0, 1]


    # Apply your optimized threshold
    prediction = (
        probability >= THRESHOLD
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Result"
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{probability:.2%}"
        )

    with result_col2:

        st.metric(
            "Operational Threshold",
            f"{THRESHOLD:.2%}"
        )


    st.progress(
        float(probability)
    )


    if prediction:

        st.error(
            "🚨 FRAUD DETECTED"
        )

        st.warning(
            f"""
            The predicted fraud probability is
            **{probability:.2%}**, which is above
            the operational threshold of
            **{THRESHOLD:.2%}**.

            This transaction should be flagged
            for further review.
            """
        )

    else:

        st.success(
            "✅ TRANSACTION APPEARS LEGITIMATE"
        )

        st.info(
            f"""
            The predicted fraud probability is
            **{probability:.2%}**, which is below
            the operational threshold of
            **{THRESHOLD:.2%}**.
            """
        )