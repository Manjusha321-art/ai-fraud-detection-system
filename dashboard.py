import os
import random
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "amount_scaler.pkl")

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

KNOWN_LOCATIONS = [
    "Hyderabad",
    "Bangalore",
    "Chennai"
]

KNOWN_DEVICES = [
    "DEVICE-001",
    "DEVICE-002"
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)

    scaler = None

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:
    model, scaler = load_artifacts()
    MODEL_READY = True
    MODEL_ERROR = None

except Exception as exc:
    model = None
    scaler = None
    MODEL_READY = False
    MODEL_ERROR = str(exc)


if "history" not in st.session_state:
    st.session_state.history = []


def make_normal_transaction(
    account_id,
    amount,
    location,
    device,
    hour
):

    transaction = {
        "account_id": account_id,
        "amount": float(amount),
        "Amount": float(amount),
        "location": location,
        "device": device,
        "device_id": device,
        "hour": int(hour),
        "Time": float(random.randint(0, 172800))
    }

    for i in range(1, 29):
        transaction[f"V{i}"] = round(
            random.uniform(-2.0, 2.0),
            6
        )

    return transaction


def make_suspicious_transaction(account_id):

    return {
        "account_id": account_id,
        "amount": 5000.0,
        "Amount": 5000.0,
        "location": "New York",
        "device": "UNKNOWN-DEVICE",
        "device_id": "UNKNOWN-DEVICE",
        "hour": 2,
        "Time": 406.0,

        "V1": -2.3122265423263,
        "V2": 1.95199201064158,
        "V3": -1.60985073229769,
        "V4": 3.99790558754602,
        "V5": -0.522187864667764,
        "V6": -1.42654531921053,
        "V7": -2.53738730624579,
        "V8": 1.39165724829804,
        "V9": -2.77008927794328,
        "V10": -2.77227214465915,
        "V11": 3.20203320709615,
        "V12": -2.89990738849473,
        "V13": -0.595221881324605,
        "V14": -4.28925378244217,
        "V15": -0.413146805280605,
        "V16": -1.47413638702157,
        "V17": -5.40063663809448,
        "V18": -0.127454693159387,
        "V19": 0.308334603984752,
        "V20": -0.0750415516800319,
        "V21": 0.155170734649284,
        "V22": 0.588841055853477,
        "V23": 0.176967718448187,
        "V24": -0.285412182616427,
        "V25": -0.628478300251405,
        "V26": -0.457413885250834,
        "V27": -0.195761272049828,
        "V28": -0.143275874698919
    }


def calculate_rule_risk(transaction):

    score = 0
    reasons = []

    amount = float(
        transaction.get(
            "amount",
            transaction.get("Amount", 0)
        )
    )

    location = transaction.get(
        "location",
        "Unknown"
    )

    device = transaction.get(
        "device_id",
        transaction.get(
            "device",
            "Unknown"
        )
    )

    hour = int(
        transaction.get(
            "hour",
            0
        )
    )

    if amount > 1000:
        score += 25
        reasons.append(
            "High transaction amount"
        )

    if device == "UNKNOWN-DEVICE":
        score += 25
        reasons.append(
            "Unknown device"
        )

    if location not in KNOWN_LOCATIONS:
        score += 25
        reasons.append(
            "Unusual location"
        )

    if hour < 6:
        score += 25
        reasons.append(
            "Unusual transaction time"
        )

    return min(score, 100), reasons


def prepare_features(transaction):

    values = {}

    for feature in FEATURE_NAMES:

        values[feature] = float(
            transaction.get(
                feature,
                0.0
            )
        )

    columns = list(FEATURE_NAMES)

    if hasattr(
        model,
        "feature_names_in_"
    ):

        columns = list(
            model.feature_names_in_
        )

        for feature in columns:

            if feature not in values:
                values[feature] = 0.0

    frame = pd.DataFrame(
        [
            [
                values[column]
                for column in columns
            ]
        ],
        columns=columns
    )

    return frame


def transform_for_model(frame):

    if scaler is None:

        return frame

    if not hasattr(
        scaler,
        "n_features_in_"
    ):

        return frame

    expected_features = int(
        scaler.n_features_in_
    )

    if expected_features == frame.shape[1]:

        transformed = scaler.transform(
            frame
        )

        return pd.DataFrame(
            transformed,
            columns=frame.columns
        )

    if (
        expected_features == 1
        and "Amount" in frame.columns
    ):

        output = frame.copy()

        output["Amount"] = (
            scaler.transform(
                output[["Amount"]]
            ).ravel()
        )

        return output

    return frame


def predict_fraud(transaction):

    frame = prepare_features(
        transaction
    )

    model_input = transform_for_model(
        frame
    )

    prediction = int(
        model.predict(
            model_input
        )[0]
    )

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            model_input
        )[0]

        classes = list(
            getattr(
                model,
                "classes_",
                range(len(probabilities))
            )
        )

        if 1 in classes:

            fraud_probability = float(
                probabilities[
                    classes.index(1)
                ]
            )

        elif len(probabilities) > 1:

            fraud_probability = float(
                probabilities[-1]
            )

        else:

            fraud_probability = float(
                probabilities[0]
            )

    else:

        fraud_probability = (
            1.0
            if prediction == 1
            else 0.0
        )

    return (
        prediction == 1,
        max(
            0.0,
            min(
                1.0,
                fraud_probability
            )
        )
    )


def severity_for_score(score):

    if score >= 75:
        return "CRITICAL", "🔴"

    if score >= 50:
        return "HIGH", "🟠"

    if score >= 25:
        return "MEDIUM", "🟡"

    return "LOW", "🟢"


st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
    }

    .subtitle {
        text-align: center;
        opacity: 0.75;
        margin-bottom: 20px;
    }

    .badge-row {
        text-align: center;
        margin-bottom: 25px;
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        margin: 4px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.3);
        font-size: 12px;
        font-weight: 700;
    }

    .pipeline {
        text-align: center;
        padding: 18px 5px;
    }

    .step {
        display: inline-block;
        padding: 10px 12px;
        margin: 4px;
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 10px;
        font-size: 11px;
        font-weight: 700;
    }

    .arrow {
        font-size: 18px;
        margin: 0 4px;
    }

    .fraud-box {
        padding: 20px;
        border-radius: 14px;
        border: 2px solid #ff4b4b;
        background: rgba(255,75,75,0.08);
    }

    .safe-box {
        padding: 20px;
        border-radius: 14px;
        border: 2px solid #00c853;
        background: rgba(0,200,83,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="main-title">
        🛡️ AI FRAUD DETECTION SYSTEM
    </div>

    <div class="subtitle">
        Real-Time Transaction Monitoring •
        Machine Learning •
        Behavioral Intelligence
    </div>

    <div class="badge-row">
        <span class="badge">
            ● SYSTEM ONLINE
        </span>

        <span class="badge">
            ◈ RANDOM FOREST ACTIVE
        </span>

        <span class="badge">
            ◇ BEHAVIORAL ENGINE ACTIVE
        </span>

        <span class="badge">
            ◆ REAL-TIME MONITORING
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


if MODEL_READY:

    st.success(
        "✅ Random Forest model and saved scaler loaded successfully."
    )

else:

    st.error(
        "❌ Model could not be loaded."
    )

    st.code(
        MODEL_ERROR or "Unknown model loading error"
    )


st.sidebar.title(
    "⚙️ CONTROL CENTER"
)

account_id = st.sidebar.text_input(
    "Account ID",
    "ACC-101"
)

transaction_type = st.sidebar.radio(
    "Transaction Type",
    [
        "Normal Transaction",
        "Suspicious Transaction",
        "Custom Transaction"
    ]
)


if transaction_type == "Custom Transaction":

    custom_amount = st.sidebar.number_input(
        "Amount (₹)",
        min_value=1.0,
        max_value=100000.0,
        value=500.0,
        step=100.0
    )

    custom_location = st.sidebar.selectbox(
        "Location",
        [
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "New York",
            "London"
        ]
    )

    custom_device = st.sidebar.selectbox(
        "Device",
        [
            "DEVICE-001",
            "DEVICE-002",
            "UNKNOWN-DEVICE"
        ]
    )

    custom_hour = st.sidebar.slider(
        "Transaction Hour",
        0,
        23,
        14
    )


analyze = st.sidebar.button(
    "🚀 ANALYZE TRANSACTION",
    use_container_width=True
)

reset = st.sidebar.button(
    "🗑️ RESET SESSION",
    use_container_width=True
)


if reset:

    st.session_state.history = []

    st.rerun()


st.header(
    "🧠 AI DETECTION PIPELINE"
)

st.markdown(
    """
    <div class="pipeline">

        <span class="step">
            TRANSACTION
        </span>

        <span class="arrow">→</span>

        <span class="step">
            FEATURE EXTRACTION
        </span>

        <span class="arrow">→</span>

        <span class="step">
            RANDOM FOREST
        </span>

        <span class="arrow">→</span>

        <span class="step">
            BEHAVIORAL ANALYSIS
        </span>

        <span class="arrow">→</span>

        <span class="step">
            RISK ENGINE
        </span>

        <span class="arrow">→</span>

        <span class="step">
            FRAUD DECISION
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


if analyze:

    if not MODEL_READY:

        st.error(
            "Model files are not available."
        )

        st.stop()


    if transaction_type == "Normal Transaction":

        transaction = make_normal_transaction(

            account_id,

            random.uniform(
                95,
                125
            ),

            random.choice(
                KNOWN_LOCATIONS
            ),

            random.choice(
                KNOWN_DEVICES
            ),

            random.randint(
                8,
                22
            )
        )

    elif transaction_type == "Suspicious Transaction":

        transaction = make_suspicious_transaction(
            account_id
        )

    else:

        transaction = make_normal_transaction(

            account_id,

            custom_amount,

            custom_location,

            custom_device,

            custom_hour
        )


    rule_score, rule_reasons = (
        calculate_rule_risk(
            transaction
        )
    )


    try:

        ml_fraud, ml_probability = (
            predict_fraud(
                transaction
            )
        )

    except Exception as exc:

        st.error(
            "ML prediction failed."
        )

        st.exception(exc)

        st.stop()


    if ml_probability >= 0.80:

        ml_risk = 90

    elif ml_probability >= 0.60:

        ml_risk = 75

    elif ml_probability >= 0.50:

        ml_risk = 60

    elif ml_probability >= 0.30:

        ml_risk = 40

    else:

        ml_risk = 0


    final_risk = max(
        rule_score,
        ml_risk
    )


    fraud_detected = bool(
        ml_fraud
        or rule_score >= 50
    )


    if fraud_detected and final_risk < 50:

        final_risk = 50


    severity, severity_icon = (
        severity_for_score(
            final_risk
        )
    )


    if fraud_detected:

        confidence = (
            ml_probability * 100
        )

    else:

        confidence = (
            (1 - ml_probability) * 100
        )


    reasons = list(
        rule_reasons
    )


    if ml_fraud:

        reasons.append(
            "Random Forest model classified the transaction as fraud"
        )


    if not reasons:

        reasons.append(
            "No major suspicious signals detected"
        )


    record = {

        "Timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "Account ID":
            account_id,

        "Amount":
            transaction["Amount"],

        "Location":
            transaction["location"],

        "Device":
            transaction["device_id"],

        "Hour":
            transaction["hour"],

        "Risk Score":
            final_risk,

        "ML Fraud Probability (%)":
            round(
                ml_probability * 100,
                2
            ),

        "ML Confidence (%)":
            round(
                confidence,
                2
            ),

        "Severity":
            severity,

        "Fraud Detected":
            fraud_detected,

        "Reasons":
            "; ".join(
                reasons
            )
    }


    st.session_state.history.append(
        record
    )


    st.divider()


    if fraud_detected:

        st.markdown(
            """
            <div class="fraud-box">

                <h2>🚨 FRAUD ALERT</h2>

                <p>
                    Suspicious transaction detected by
                    the hybrid fraud detection system.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="safe-box">

                <h2>🟢 TRANSACTION APPROVED</h2>

                <p>
                    No major fraud indicators detected.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.subheader(
        "💳 Latest Transaction Analysis"
    )


    c1, c2, c3, c4, c5 = st.columns(5)


    c1.metric(
        "Amount",
        f"₹{transaction['Amount']:,.2f}"
    )


    c2.metric(
        "Risk Score",
        f"{final_risk}/100"
    )


    c3.metric(
        "ML Probability",
        f"{ml_probability * 100:.2f}%"
    )


    c4.metric(
        "ML Confidence",
        f"{confidence:.2f}%"
    )


    c5.metric(
        "Severity",
        f"{severity_icon} {severity}"
    )


    st.subheader(
        "🔍 Risk Breakdown"
    )


    amount_risk = (
        25
        if transaction["Amount"] > 1000
        else 0
    )


    device_risk = (
        25
        if transaction["device_id"]
        == "UNKNOWN-DEVICE"
        else 0
    )


    location_risk = (
        25
        if transaction["location"]
        not in KNOWN_LOCATIONS
        else 0
    )


    time_risk = (
        25
        if int(transaction["hour"]) < 6
        else 0
    )


    r1, r2, r3, r4 = st.columns(4)


    r1.metric(
        "Amount Anomaly",
        f"+{amount_risk}"
    )


    r2.metric(
        "Unknown Device",
        f"+{device_risk}"
    )


    r3.metric(
        "Unusual Location",
        f"+{location_risk}"
    )


    r4.metric(
        "Unusual Time",
        f"+{time_risk}"
    )


    st.progress(
        min(
            final_risk / 100,
            1.0
        )
    )


    st.subheader(
        "📋 Detection Reasons"
    )


    for reason in reasons:

        st.write(
            f"🔴 {reason}"
        )


    with st.expander(
        "🔎 Full Transaction Details"
    ):

        d1, d2 = st.columns(2)


        with d1:

            st.write(
                f"**Account ID:** "
                f"{transaction['account_id']}"
            )

            st.write(
                f"**Amount:** "
                f"₹{transaction['Amount']:,.2f}"
            )

            st.write(
                f"**Location:** "
                f"{transaction['location']}"
            )

            st.write(
                f"**Device:** "
                f"{transaction['device_id']}"
            )


        with d2:

            st.write(
                f"**Hour:** "
                f"{transaction['hour']}"
            )

            st.write(
                f"**Time Feature:** "
                f"{transaction['Time']}"
            )

            st.write(
                "**ML Features:** 30"
            )

            st.write(
                "**Model:** Random Forest"
            )


st.divider()

st.header(
    "📡 SYSTEM TELEMETRY"
)


history = st.session_state.history


total = len(history)


fraud_count = sum(
    row["Fraud Detected"]
    for row in history
)


normal_count = (
    total - fraud_count
)


fraud_rate = (
    fraud_count / total * 100
    if total
    else 0
)


average_risk = (
    sum(
        row["Risk Score"]
        for row in history
    ) / total
    if total
    else 0
)


t1, t2, t3, t4, t5 = st.columns(5)


t1.metric(
    "SYSTEM STATUS",
    "ONLINE"
)


t2.metric(
    "TRANSACTIONS",
    total
)


t3.metric(
    "🚨 FRAUD EVENTS",
    fraud_count
)


t4.metric(
    "✅ NORMAL EVENTS",
    normal_count
)


t5.metric(
    "FRAUD RATE",
    f"{fraud_rate:.1f}%"
)


st.divider()

st.header(
    "📋 TRANSACTION HISTORY"
)


if history:

    history_df = pd.DataFrame(
        history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No transactions analyzed yet."
    )


if history:

    st.subheader(
        "📈 Risk Score Trend"
    )


    chart_df = pd.DataFrame({

        "Transaction":
            range(
                1,
                len(history) + 1
            ),

        "Risk Score":
            [
                row["Risk Score"]
                for row in history
            ]
    })


    st.line_chart(
        chart_df,
        x="Transaction",
        y="Risk Score",
        use_container_width=True
    )


st.divider()

st.header(
    "🎯 What-If Fraud Attack Simulator"
)


st.write(
    "Change transaction conditions and observe the rule-based risk response."
)


s1, s2 = st.columns(2)


with s1:

    sim_amount = st.slider(
        "💰 Amount (₹)",
        0,
        10000,
        500,
        50
    )


    sim_location = st.selectbox(
        "📍 Location",
        [
            "Hyderabad",
            "Bangalore",
            "Chennai",
            "New York",
            "London"
        ]
    )


with s2:

    sim_device = st.selectbox(
        "💻 Device",
        [
            "DEVICE-001",
            "DEVICE-002",
            "UNKNOWN-DEVICE"
        ]
    )


    sim_hour = st.slider(
        "🕒 Hour",
        0,
        23,
        14
    )


sim_transaction = {

    "amount":
        sim_amount,

    "location":
        sim_location,

    "device_id":
        sim_device,

    "hour":
        sim_hour
}


sim_risk, sim_reasons = (
    calculate_rule_risk(
        sim_transaction
    )
)


sim_severity, sim_icon = (
    severity_for_score(
        sim_risk
    )
)


q1, q2, q3 = st.columns(3)


q1.metric(
    "Simulated Risk",
    f"{sim_risk}/100"
)


q2.metric(
    "Severity",
    f"{sim_icon} {sim_severity}"
)


q3.metric(
    "Signals",
    len(sim_reasons)
)


for reason in sim_reasons:

    st.write(
        f"🔴 +25 — {reason}"
    )


if not sim_reasons:

    st.success(
        "No major suspicious signals."
    )


st.divider()

st.header(
    "📈 Fraud Attack Progression"
)


p1, p2, p3, p4 = st.columns(4)


with p1:

    st.success(
        "🟢 STAGE 1\n\n"
        "Normal Activity\n\n"
        "Known device • Normal location"
    )


with p2:

    st.warning(
        "🟡 STAGE 2\n\n"
        "Minor Behavioral Change\n\n"
        "Amount deviation"
    )


with p3:

    st.warning(
        "🟠 STAGE 3\n\n"
        "Suspicious Behavior\n\n"
        "New device • New location • Unusual time"
    )


with p4:

    st.error(
        "🔴 STAGE 4\n\n"
        "Possible Account Takeover\n\n"
        "Multiple anomalies"
    )


if sim_risk < 25:

    current_stage = (
        "Stage 1 — Normal Activity"
    )

elif sim_risk < 50:

    current_stage = (
        "Stage 2 — Minor Behavioral Change"
    )

elif sim_risk < 75:

    current_stage = (
        "Stage 3 — Suspicious Behavior"
    )

else:

    current_stage = (
        "Stage 4 — Possible Account Takeover"
    )


st.info(
    f"Current simulated stage: "
    f"**{current_stage}**"
)


if history:

    st.divider()

    st.header(
        "🧬 Account Behavioral Fingerprint"
    )


    amounts = [
        float(row["Amount"])
        for row in history
    ]


    locations = sorted(
        {
            row["Location"]
            for row in history
        }
    )


    devices = sorted(
        {
            row["Device"]
            for row in history
        }
    )


    hours = sorted(
        {
            int(row["Hour"])
            for row in history
        }
    )


    b1, b2, b3, b4, b5 = st.columns(5)


    b1.metric(
        "Average Amount",
        f"₹{sum(amounts) / len(amounts):,.2f}"
    )


    b2.metric(
        "Maximum Amount",
        f"₹{max(amounts):,.2f}"
    )


    b3.metric(
        "Locations",
        len(locations)
    )


    b4.metric(
        "Devices",
        len(devices)
    )


    b5.metric(
        "Average Risk",
        f"{average_risk:.1f}"
    )


    st.write(
        "**📍 Locations:** "
        + ", ".join(locations)
    )


    st.write(
        "**💻 Devices:** "
        + ", ".join(devices)
    )


    st.write(
        "**🕒 Observed Hours:** "
        + ", ".join(
            str(hour)
            for hour in hours
        )
    )


st.divider()

st.header(
    "📥 Export & Investigation"
)


if history:

    export_df = pd.DataFrame(
        history
    )


    csv_data = export_df.to_csv(
        index=False
    )


    e1, e2 = st.columns(2)


    with e1:

        st.download_button(
            "📊 Download Transaction CSV",
            data=csv_data,
            file_name="fraud_transaction_report.csv",
            mime="text/csv",
            use_container_width=True
        )


    latest = history[-1]


    report = f"""
AI FRAUD DETECTION SYSTEM
FRAUD INVESTIGATION REPORT
==============================================

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

TRANSACTION DETAILS
----------------------------------------------
Account ID: {latest["Account ID"]}
Amount: ₹{latest["Amount"]:,.2f}
Location: {latest["Location"]}
Device: {latest["Device"]}
Hour: {latest["Hour"]}

AI ANALYSIS
----------------------------------------------
Risk Score: {latest["Risk Score"]}/100
ML Fraud Probability: {latest["ML Fraud Probability (%)"]}%
ML Confidence: {latest["ML Confidence (%)"]}%
Severity: {latest["Severity"]}
Fraud Detected: {"YES" if latest["Fraud Detected"] else "NO"}

DETECTION REASONS
----------------------------------------------
{latest["Reasons"]}

MODEL
----------------------------------------------
Random Forest
30 ML Features
Hybrid ML + Rule-Based Detection
Behavioral Intelligence

==============================================
"""


    with e2:

        st.download_button(
            "📄 Download Investigation Report",
            data=report,
            file_name="fraud_investigation_report.txt",
            mime="text/plain",
            use_container_width=True
        )

else:

    st.info(
        "Analyze a transaction to enable downloads."
    )


st.divider()

st.header(
    "🔐 Model & Feature Architecture"
)


a1, a2, a3, a4 = st.columns(4)


a1.metric(
    "Model",
    "Random Forest"
)


a2.metric(
    "Features",
    "30"
)


a3.metric(
    "Detection",
    "Hybrid"
)


a4.metric(
    "Decision",
    "ML + Rules"
)


st.write(
    "Transaction → Feature Extraction → "
    "Random Forest → Behavioral Analysis → "
    "Risk Engine → Fraud Decision"
)


st.caption(
    "AI FRAUD DETECTION SYSTEM • "
    f"Transactions: {total} • "
    f"Fraud Events: {fraud_count} • "
    "Status: ONLINE"
)
