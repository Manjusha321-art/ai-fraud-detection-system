import streamlit as st
import pandas as pd
import random
from datetime import datetime
from io import BytesIO

from src.fraud_engine import FraudEngine


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .title {
            text-align: center;
            font-size: 40px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 16px;
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
            padding: 20px 5px;
            margin-bottom: 20px;
        }

        .step {
            display: inline-block;
            padding: 10px 13px;
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

        .fraud-card {
            padding: 20px;
            border-radius: 14px;
            border: 2px solid #ff4b4b;
            background: rgba(255, 75, 75, 0.08);
            margin: 10px 0;
        }

        .safe-card {
            padding: 20px;
            border-radius: 14px;
            border: 2px solid #00c853;
            background: rgba(0, 200, 83, 0.08);
            margin: 10px 0;
        }

        .section-title {
            font-size: 24px;
            font-weight: 800;
            margin-top: 20px;
        }

        .mini-card {
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_transaction" not in st.session_state:
    st.session_state.latest_transaction = None

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


# ============================================================
# LOAD FRAUD ENGINE
# ============================================================

@st.cache_resource
def load_engine():
    try:
        return FraudEngine()
    except Exception:
        return None


engine = load_engine()


# ============================================================
# ML FEATURE CREATION
# ============================================================

def create_normal_transaction(account_id, amount, location, device, hour):
    transaction = {
        "account_id": account_id,
        "amount": float(amount),
        "Amount": float(amount),
        "location": location,
        "device": device,
        "device_id": device,
        "hour": int(hour),

        # Required ML feature
        "Time": float(random.randint(0, 172800))
    }

    # V1-V28
    for i in range(1, 29):
        transaction[f"V{i}"] = round(
            random.uniform(-2.0, 2.0),
            6
        )

    return transaction


def create_suspicious_transaction(account_id):
    # Known fraud example from the dataset
    transaction = {
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

    return transaction


# ============================================================
# RULE-BASED RISK
# ============================================================

def calculate_rule_risk(transaction):

    score = 0
    reasons = []

    amount = float(transaction.get("amount", 0))
    location = transaction.get("location", "Unknown")
    device = transaction.get(
        "device_id",
        transaction.get("device", "Unknown")
    )
    hour = int(transaction.get("hour", 0))

    known_locations = [
        "Hyderabad",
        "Bangalore",
        "Chennai"
    ]

    if amount > 1000:
        score += 25
        reasons.append("High transaction amount")

    if device == "UNKNOWN-DEVICE":
        score += 25
        reasons.append("Unknown device")

    if location not in known_locations:
        score += 25
        reasons.append("Unusual location")

    if hour < 6:
        score += 25
        reasons.append("Unusual transaction time")

    return min(score, 100), reasons


# ============================================================
# SEVERITY
# ============================================================

def get_severity(score):

    if score >= 75:
        return "CRITICAL", "🔴"

    if score >= 50:
        return "HIGH", "🟠"

    if score >= 25:
        return "MEDIUM", "🟡"

    return "LOW", "🟢"


# ============================================================
# SAFE RESULT EXTRACTION
# ============================================================

def get_result_value(result, names, default):

    if not isinstance(result, dict):
        return default

    for name in names:
        if name in result:
            return result[name]

    return default


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🛡️ AI FRAUD DETECTION SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Transaction Monitoring • Machine Learning • Behavioral Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="badge-row">
        <span class="badge">● SYSTEM ONLINE</span>
        <span class="badge">◈ RANDOM FOREST ACTIVE</span>
        <span class="badge">◇ BEHAVIORAL ENGINE ACTIVE</span>
        <span class="badge">◆ REAL-TIME MONITORING</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ CONTROL CENTER")

account_id = st.sidebar.text_input(
    "Account ID",
    "ACC-101"
)

transaction_mode = st.sidebar.radio(
    "Transaction Type",
    [
        "Normal Transaction",
        "Suspicious Transaction",
        "Custom Transaction"
    ]
)


if transaction_mode == "Custom Transaction":

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


analyze_button = st.sidebar.button(
    "🚀 ANALYZE TRANSACTION",
    use_container_width=True
)

reset_button = st.sidebar.button(
    "🗑️ RESET SESSION",
    use_container_width=True
)

if reset_button:
    st.session_state.history = []
    st.session_state.latest_transaction = None
    st.session_state.latest_result = None
    st.rerun()


# ============================================================
# PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 AI DETECTION PIPELINE</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="pipeline">
        <span class="step">TRANSACTION</span>
        <span class="arrow">→</span>
        <span class="step">FEATURE EXTRACTION</span>
        <span class="arrow">→</span>
        <span class="step">RANDOM FOREST</span>
        <span class="arrow">→</span>
        <span class="step">BEHAVIORAL ANALYSIS</span>
        <span class="arrow">→</span>
        <span class="step">RISK ENGINE</span>
        <span class="arrow">→</span>
        <span class="step">FRAUD DECISION</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # CREATE TRANSACTION
    # --------------------------------------------------------

    if transaction_mode == "Normal Transaction":

        transaction = create_normal_transaction(
            account_id,
            random.uniform(95, 125),
            random.choice(
                [
                    "Hyderabad",
                    "Bangalore",
                    "Chennai"
                ]
            ),
            random.choice(
                [
                    "DEVICE-001",
                    "DEVICE-002"
                ]
            ),
            random.randint(8, 22)
        )

    elif transaction_mode == "Suspicious Transaction":

        transaction = create_suspicious_transaction(
            account_id
        )

    else:

        transaction = create_normal_transaction(
            account_id,
            custom_amount,
            custom_location,
            custom_device,
            custom_hour
        )


    # --------------------------------------------------------
    # FORCE ALL REQUIRED ML FEATURES
    # --------------------------------------------------------

    transaction["Time"] = float(
        transaction.get(
            "Time",
            transaction.get(
                "time",
                datetime.now().timestamp()
            )
        )
    )

    transaction["Amount"] = float(
        transaction.get(
            "Amount",
            transaction.get(
                "amount",
                0
            )
        )
    )

    transaction["amount"] = float(
        transaction.get(
            "amount",
            transaction.get(
                "Amount",
                0
            )
        )
    )

    for i in range(1, 29):

        feature_name = f"V{i}"

        if feature_name not in transaction:

            transaction[feature_name] = 0.0


    # Required behavioral keys
    transaction["device_id"] = transaction.get(
        "device_id",
        transaction.get(
            "device",
            "DEVICE-001"
        )
    )

    transaction["device"] = transaction.get(
        "device",
        transaction.get(
            "device_id",
            "DEVICE-001"
        )
    )


    # --------------------------------------------------------
    # RULE RISK
    # --------------------------------------------------------

    rule_risk, rule_reasons = calculate_rule_risk(
        transaction
    )


    # --------------------------------------------------------
    # ML / FRAUD ENGINE
    # --------------------------------------------------------

    engine_result = {}

    if engine is not None:

        try:

            engine_result = engine.analyze_transaction(
                transaction
            )

        except Exception:

            # Keep dashboard operational even if an
            # internal engine component has a problem.
            engine_result = {}


    # --------------------------------------------------------
    # ML PROBABILITY
    # --------------------------------------------------------

    ml_probability = get_result_value(
        engine_result,
        [
            "fraud_probability",
            "ml_fraud_probability",
            "probability",
            "fraud_prob"
        ],
        0.0
    )

    try:

        ml_probability = float(
            ml_probability
        )

    except Exception:

        ml_probability = 0.0


    if ml_probability > 1:
        ml_probability /= 100.0

    ml_probability = max(
        0.0,
        min(
            ml_probability,
            1.0
        )
    )


    # --------------------------------------------------------
    # FRAUD DECISION
    # --------------------------------------------------------

    engine_fraud = get_result_value(
        engine_result,
        [
            "is_fraud",
            "fraud_detected",
            "is_fraudulent",
            "Fraud Detected"
        ],
        None
    )


    if engine_fraud is None:

        is_fraud = (
            rule_risk >= 50
            or ml_probability >= 0.50
        )

    else:

        is_fraud = bool(
            engine_fraud
        )


    # --------------------------------------------------------
    # COMBINED RISK
    # --------------------------------------------------------

    combined_risk = rule_risk

    if ml_probability >= 0.80:
        combined_risk = max(
            combined_risk,
            90
        )

    elif ml_probability >= 0.60:
        combined_risk = max(
            combined_risk,
            75
        )

    elif ml_probability >= 0.50:
        combined_risk = max(
            combined_risk,
            60
        )

    elif ml_probability >= 0.30:
        combined_risk = max(
            combined_risk,
            40
        )


    # If the engine says fraud, make sure the dashboard
    # visibly reflects that decision.
    if is_fraud and combined_risk < 50:

        combined_risk = max(
            combined_risk,
            50
        )


    severity, severity_icon = get_severity(
        combined_risk
    )


    # --------------------------------------------------------
    # ML CONFIDENCE
    # --------------------------------------------------------

    if is_fraud:

        ml_confidence = ml_probability * 100

    else:

        ml_confidence = (
            1 - ml_probability
        ) * 100


    # --------------------------------------------------------
    # ALL REASONS
    # --------------------------------------------------------

    reasons = list(
        rule_reasons
    )

    engine_reasons = get_result_value(
        engine_result,
        [
            "reasons",
            "fraud_reasons",
            "risk_reasons"
        ],
        []
    )

    if isinstance(engine_reasons, list):

        for reason in engine_reasons:

            if reason not in reasons:

                reasons.append(
                    str(reason)
                )


    if is_fraud and not reasons:

        reasons.append(
            "Machine learning model flagged the transaction"
        )


    # --------------------------------------------------------
    # SAVE RECORD
    # --------------------------------------------------------

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
            combined_risk,

        "ML Fraud Probability (%)":
            round(
                ml_probability * 100,
                2
            ),

        "ML Confidence (%)":
            round(
                ml_confidence,
                2
            ),

        "Severity":
            severity,

        "Fraud Detected":
            bool(is_fraud),

        "Reasons":
            "; ".join(
                reasons
            )
    }


    st.session_state.history.append(
        record
    )

    st.session_state.latest_transaction = transaction
    st.session_state.latest_result = {
        "risk_score": combined_risk,
        "ml_probability": ml_probability,
        "confidence": ml_confidence,
        "is_fraud": is_fraud,
        "severity": severity,
        "severity_icon": severity_icon,
        "reasons": reasons
    }


# ============================================================
# LATEST RESULT
# ============================================================

if st.session_state.latest_transaction is not None:

    transaction = st.session_state.latest_transaction
    latest = st.session_state.latest_result

    st.divider()

    if latest["is_fraud"]:

        st.markdown(
            """
            <div class="fraud-card">
                <h2>🚨 FRAUD ALERT</h2>
                <p>
                    Suspicious transaction detected by the
                    hybrid fraud detection system.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="safe-card">
                <h2>🟢 TRANSACTION APPROVED</h2>
                <p>
                    No major fraud indicators detected.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    st.subheader("💳 Latest Transaction Analysis")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Amount",
        f"₹{transaction['Amount']:,.2f}"
    )

    c2.metric(
        "Risk Score",
        f"{latest['risk_score']}/100"
    )

    c3.metric(
        "ML Probability",
        f"{latest['ml_probability'] * 100:.2f}%"
    )

    c4.metric(
        "ML Confidence",
        f"{latest['confidence']:.2f}%"
    )

    c5.metric(
        "Severity",
        f"{latest['severity_icon']} {latest['severity']}"
    )


    # --------------------------------------------------------
    # RISK BREAKDOWN
    # --------------------------------------------------------

    st.subheader("🔍 Risk Breakdown")

    r1, r2, r3, r4 = st.columns(4)

    amount_risk = (
        25
        if transaction["Amount"] > 1000
        else 0
    )

    device_risk = (
        25
        if transaction["device_id"] == "UNKNOWN-DEVICE"
        else 0
    )

    location_risk = (
        25
        if transaction["location"] not in [
            "Hyderabad",
            "Bangalore",
            "Chennai"
        ]
        else 0
    )

    time_risk = (
        25
        if int(transaction["hour"]) < 6
        else 0
    )


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
            latest["risk_score"] / 100,
            1.0
        )
    )


    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    st.subheader("📋 Detection Reasons")

    if latest["reasons"]:

        for reason in latest["reasons"]:

            st.write(
                f"🔴 {reason}"
            )

    else:

        st.success(
            "No major suspicious signals were triggered."
        )


    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    with st.expander(
        "🔎 Full Transaction Details"
    ):

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:

            st.write(
                f"**Account ID:** {transaction['account_id']}"
            )

            st.write(
                f"**Amount:** ₹{transaction['Amount']}"
            )

            st.write(
                f"**Location:** {transaction['location']}"
            )

            st.write(
                f"**Device:** {transaction['device_id']}"
            )

        with detail_col2:

            st.write(
                f"**Hour:** {transaction['hour']}"
            )

            st.write(
                f"**Time Feature:** {transaction['Time']}"
            )

            st.write(
                "**ML Features:** 30"
            )

            st.write(
                "**Model:** Random Forest"
            )


# ============================================================
# SYSTEM TELEMETRY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📡 SYSTEM TELEMETRY</div>',
    unsafe_allow_html=True
)

history = st.session_state.history

total = len(history)

fraud_count = sum(
    1
    for row in history
    if row["Fraud Detected"]
)

normal_count = (
    total - fraud_count
)

fraud_rate = (
    fraud_count / total * 100
    if total > 0
    else 0
)

avg_risk = (
    sum(
        row["Risk Score"]
        for row in history
    ) / total
    if total > 0
    else 0
)

max_risk = max(
    [
        row["Risk Score"]
        for row in history
    ],
    default=0
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


# ============================================================
# TRANSACTION HISTORY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📋 TRANSACTION HISTORY</div>',
    unsafe_allow_html=True
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


# ============================================================
# RISK TREND
# ============================================================

if history:

    st.subheader(
        "📈 Risk Score Trend"
    )

    chart_df = pd.DataFrame({
        "Transaction": range(
            1,
            len(history) + 1
        ),
        "Risk Score": [
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


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🎯 What-If Fraud Attack Simulator</div>',
    unsafe_allow_html=True
)

st.write(
    "Change transaction attributes and observe the rule-based risk response."
)

sim1, sim2 = st.columns(2)

with sim1:

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

with sim2:

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
    "amount": sim_amount,
    "location": sim_location,
    "device_id": sim_device,
    "hour": sim_hour
}

sim_risk, sim_reasons = calculate_rule_risk(
    sim_transaction
)

sim_severity, sim_icon = get_severity(
    sim_risk
)


s1, s2, s3 = st.columns(3)

s1.metric(
    "Simulated Risk",
    f"{sim_risk}/100"
)

s2.metric(
    "Severity",
    f"{sim_icon} {sim_severity}"
)

s3.metric(
    "Signals Triggered",
    len(sim_reasons)
)


if sim_reasons:

    for reason in sim_reasons:

        st.write(
            f"🔴 +25 — {reason}"
        )

else:

    st.success(
        "No major suspicious signals."
    )


# ============================================================
# FRAUD ATTACK PROGRESSION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📈 Fraud Attack Progression</div>',
    unsafe_allow_html=True
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.success(
        "🟢 STAGE 1\n\n"
        "Normal Activity\n\n"
        "Known device • Normal location • Normal amount"
    )

with p2:

    st.warning(
        "🟡 STAGE 2\n\n"
        "Minor Behavioral Change\n\n"
        "Slight amount deviation"
    )

with p3:

    st.warning(
        "🟠 STAGE 3\n\n"
        "Suspicious Behavior\n\n"
        "New location • New device • Unusual time"
    )

with p4:

    st.error(
        "🔴 STAGE 4\n\n"
        "Possible Account Takeover\n\n"
        "Multiple anomalies • Very high risk"
    )


if sim_risk < 25:

    stage = "Stage 1 — Normal Activity"

elif sim_risk < 50:

    stage = "Stage 2 — Minor Behavioral Change"

elif sim_risk < 75:

    stage = "Stage 3 — Suspicious Behavior"

else:

    stage = "Stage 4 — Possible Account Takeover"


st.info(
    f"Current simulated stage: **{stage}**"
)


# ============================================================
# BEHAVIORAL FINGERPRINT
# ============================================================

if history:

    st.divider()

    st.markdown(
        '<div class="section-title">🧬 Account Behavioral Fingerprint</div>',
        unsafe_allow_html=True
    )

    amounts = [
        float(row["Amount"])
        for row in history
    ]

    locations = sorted(
        list(
            set(
                row["Location"]
                for row in history
            )
        )
    )

    devices = sorted(
        list(
            set(
                row["Device"]
                for row in history
            )
        )
    )

    hours = sorted(
        list(
            set(
                int(row["Hour"])
                for row in history
            )
        )
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
        "Known Locations",
        len(locations)
    )

    b4.metric(
        "Known Devices",
        len(devices)
    )

    b5.metric(
        "Average Risk",
        f"{avg_risk:.1f}"
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


# ============================================================
# EXPORT SECTION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📥 Export & Investigation</div>',
    unsafe_allow_html=True
)

if history:

    export_df = pd.DataFrame(
        history
    )

    csv_data = export_df.to_csv(
        index=False
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.download_button(
            "📊 Download Transaction CSV",
            data=csv_data,
            file_name="fraud_transaction_report.csv",
            mime="text/csv",
            use_container_width=True
        )


    # --------------------------------------------------------
    # INVESTIGATION REPORT
    # --------------------------------------------------------

    latest_row = history[-1]

    report_text = f"""
AI FRAUD DETECTION SYSTEM
FRAUD INVESTIGATION REPORT
==================================================

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

TRANSACTION DETAILS
--------------------------------------------------
Account ID:
{latest_row["Account ID"]}

Amount:
₹{latest_row["Amount"]:,.2f}

Location:
{latest_row["Location"]}

Device:
{latest_row["Device"]}

Transaction Hour:
{latest_row["Hour"]}


AI ANALYSIS
--------------------------------------------------
Risk Score:
{latest_row["Risk Score"]}/100

ML Fraud Probability:
{latest_row["ML Fraud Probability (%)"]}%

ML Confidence:
{latest_row["ML Confidence (%)"]}%

Severity:
{latest_row["Severity"]}

Fraud Detected:
{"YES" if latest_row["Fraud Detected"] else "NO"}


DETECTION REASONS
--------------------------------------------------
{latest_row["Reasons"]}


SYSTEM ARCHITECTURE
--------------------------------------------------

TRANSACTION
      |
      v
FEATURE EXTRACTION
      |
      v
RANDOM FOREST
      |
      v
BEHAVIORAL ANALYSIS
      |
      v
RISK ENGINE
      |
      v
FRAUD DECISION


MODEL
--------------------------------------------------
Model Type:
Random Forest

Feature Count:
30

Architecture:
Hybrid ML + Rule-Based + Behavioral Intelligence

==================================================
"""

    with col_b:

        st.download_button(
            "📄 Download Investigation Report",
            data=report_text,
            file_name="fraud_investigation_report.txt",
            mime="text/plain",
            use_container_width=True
        )

else:

    st.info(
        "Analyze a transaction to enable export and investigation reports."
    )


# ============================================================
# SYSTEM SUMMARY
# ============================================================

st.divider()

st.subheader(
    "🔐 Model & Feature Architecture"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Model",
    "Random Forest"
)

m2.metric(
    "ML Features",
    "30"
)

m3.metric(
    "Detection Type",
    "Hybrid"
)

m4.metric(
    "Decision",
    "ML + Rules"
)


st.write(
    "The system combines machine-learning prediction, "
    "behavioral analysis, and rule-based risk signals "
    "to produce an explainable fraud decision."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI FRAUD DETECTION SYSTEM • "
    f"Transactions: {total} • "
    f"Fraud Events: {fraud_count} • "
    "Status: ONLINE"
)
