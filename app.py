import random
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference import OlistCustomerIntelligence

st.set_page_config(
    page_title="Olist Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
COLOR_SEGMENT = "#1F8A8C"   # teal  - "who is this customer"
COLOR_CHURN = "#E0562B"     # burnt orange/coral - "will we lose them"
COLOR_REPEAT = "#D9A441"    # mango/amber - "will they come back"
COLOR_INK = "#101B2D"       # deep navy - headers, sidebar
COLOR_CANVAS = "#F7F8FA"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

html, body, [class*="css"]  {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }}

[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}

.section-card {{
    background: linear-gradient(180deg, color-mix(in srgb, var(--accent, {COLOR_SEGMENT}) 6%, white) 0%, white 65%);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #E7E9EE;
    border-left: 5px solid var(--accent, {COLOR_SEGMENT});
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}}
.section-card:hover {{
    box-shadow: 0 6px 20px rgba(16, 27, 45, 0.08);
    transform: translateY(-1px);
}}
.badge {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    animation: pop 0.35s ease;
}}
@keyframes pop {{
    0%   {{ transform: scale(0.6); opacity: 0; }}
    70%  {{ transform: scale(1.08); opacity: 1; }}
    100% {{ transform: scale(1); }}
}}
.badge-green {{ background:#E4F5EA; color:#1D7A3E; }}
.badge-amber {{ background:#FCF0DA; color:#9A6300; }}
.badge-red   {{ background:#FBE4DE; color:#B23A17; }}
.badge-teal  {{ background:#E1F1F1; color:{COLOR_SEGMENT}; }}

.gradient-title {{
    background: linear-gradient(90deg, {COLOR_SEGMENT}, {COLOR_CHURN} 55%, {COLOR_REPEAT});
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}}

div.stButton > button {{
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
div.stButton > button:hover {{
    transform: translateY(-1px) scale(1.02);
    box-shadow: 0 4px 12px rgba(16, 27, 45, 0.15);
}}

.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7280;
}}
</style>
""", unsafe_allow_html=True)


def badge(text, kind):
    return f'<span class="badge badge-{kind}">{text}</span>'


def risk_kind(label):
    return {"Low": "green", "Medium": "amber", "High": "red"}.get(label, "teal")


def gauge(value, accent):
    """0-1 probability rendered as a colored gauge with a numeric readout."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": accent, "thickness": 0.3},
            "steps": [
                {"range": [0, 30], "color": "#E4F5EA"},
                {"range": [30, 60], "color": "#FCF0DA"},
                {"range": [60, 100], "color": "#FBE4DE"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def animate_gauge(placeholder, target_value, accent, frames=10, total_seconds=0.5):
    """Fills the gauge from 0 up to target_value instead of popping in instantly."""
    for i in range(1, frames + 1):
        placeholder.plotly_chart(
            gauge(target_value * i / frames, accent),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        time.sleep(total_seconds / frames)


SEGMENT_SPINNER_MSGS = [
    "Crunching recency, frequency and spend...",
    "Placing the customer on the map...",
    "Comparing against segment centroids...",
]
CHURN_SPINNER_MSGS = [
    "Reading the behavioural tea leaves...",
    "Weighing delivery, reviews and spend...",
    "Consulting the Gradient Boosting model...",
]
REPEAT_SPINNER_MSGS = [
    "Sizing up the first order...",
    "Checking what first-time buyers usually do...",
    "Asking the SVM for its verdict...",
]


FEATURE_LABELS = {
    "frequency": "Order frequency",
    "monetary_log": "Total spend (log)",
    "avg_review_score": "Avg review score",
    "avg_delivery_days": "Avg delivery days",
    "on_time_rate": "On-time delivery rate",
    "avg_payment_installments": "Avg payment installments",
}

# ----------------------------------------------------------------------------
# Load models
# ----------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    return OlistCustomerIntelligence()

try:
    with st.spinner("Waking up the models..."):
        engine = load_engine()
except FileNotFoundError as e:
    st.error(
        "Model files are missing. Run `Team3_Deployment_NextSteps.ipynb` on the "
        "real dataset, then copy everything from its `deployment_models/` output "
        "folder into this repo alongside app.py.\n\n"
        f"Missing: {e}"
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

PRESETS = {
    "Loyal regular": dict(recency=15, frequency=5, monetary=620.0),
    "New customer": dict(recency=5, frequency=1, monetary=140.0),
    "At-risk / dormant": dict(recency=240, frequency=1, monetary=90.0),
}

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛍️ Olist Customer Intelligence")
    st.caption("Team 3 · Customer Segmentation and Purchase Behavior Analysis")

    st.markdown("**Try an example customer**")
    for name, vals in PRESETS.items():
        if st.button(name, use_container_width=True):
            st.session_state["recency"] = vals["recency"]
            st.session_state["frequency"] = vals["frequency"]
            st.session_state["monetary"] = vals["monetary"]
            st.session_state["f2"] = vals["frequency"]
            st.session_state["m2"] = vals["monetary"]

    with st.expander("About the models"):
        st.markdown(f"""
{badge("Segment", "teal")} **K-Means, k=4** — groups a customer by recency,
frequency and monetary value.

{badge("Churn risk", "red")} **Gradient Boosting** — probability of no
purchase in the next 180 days, learned from behaviour (not recency itself,
to avoid leaking the label).

{badge("Repeat purchase", "amber")} **SVM** — probability a first-time buyer
returns, using only what's known right after checkout.

Agglomerative Clustering and the frequency==1 Random Forest churn model were
also trained and are documented in the analysis notebooks, but aren't wired
into this live app.
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Retail &amp; Customer Analytics</div>', unsafe_allow_html=True)
st.markdown('<h1 class="gradient-title">Olist Customer Intelligence</h1>', unsafe_allow_html=True)
st.caption(
    "Pick a tab below, enter a customer profile (or load an example from the "
    "sidebar), and get a live prediction from the trained model."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🧭 Segment", "⚠️ Churn risk", "🔁 Repeat purchase", "🕘 History",
])

# ----------------------------------------------------------------------------
# Tab 1 — Segment
# ----------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="section-card" style="--accent:%s">' % COLOR_SEGMENT,
                     unsafe_allow_html=True)
        st.subheader("Customer profile")
        recency = st.number_input("Days since last order", min_value=0, value=30, key="recency")
        frequency = st.number_input("Number of orders placed", min_value=1, value=1, key="frequency")
        monetary = st.number_input("Total amount spent (R$)", min_value=0.0, value=150.0, key="monetary")
        run_seg = st.button("Predict segment", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if run_seg:
            with st.spinner(random.choice(SEGMENT_SPINNER_MSGS)):
                time.sleep(0.4)
                result = engine.predict_segment(recency, frequency, monetary)
                profile = engine.get_segment_profile(result["cluster_id"])

            st.markdown('<div class="section-card" style="--accent:%s">' % COLOR_SEGMENT,
                         unsafe_allow_html=True)
            st.markdown(f"### {badge(result['segment_name'], 'teal')}", unsafe_allow_html=True)
            st.caption("How this customer compares to the average customer in this segment")

            c1, c2, c3 = st.columns(3)
            c1.metric("Recency (days)", f"{recency:.0f}",
                      delta=f"{recency - profile['recency']:+.0f} vs segment avg",
                      delta_color="inverse")
            c2.metric("Orders", f"{frequency:.0f}",
                      delta=f"{frequency - profile['frequency']:+.1f} vs segment avg")
            c3.metric("Spend (R$)", f"{monetary:,.0f}",
                      delta=f"{monetary - profile['monetary']:+,.0f} vs segment avg")
            st.markdown('</div>', unsafe_allow_html=True)

            st.toast(f"Segment: {result['segment_name']}", icon="🧭")
            if result["segment_name"] == "Loyal High-Value":
                st.balloons()

            st.session_state.history.append({
                "Model": "Segment", "Input": f"R={recency}, F={frequency}, M={monetary}",
                "Result": result["segment_name"],
            })
        else:
            st.info("Fill in the profile and click **Predict segment** to see results here.")

# ----------------------------------------------------------------------------
# Tab 2 — Churn risk
# ----------------------------------------------------------------------------
with tab2:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="section-card" style="--accent:%s">' % COLOR_CHURN,
                     unsafe_allow_html=True)
        st.subheader("Customer behaviour")
        st.caption("Churn = no purchase in the last 180 days")
        frequency2 = st.number_input("Number of orders placed", min_value=1, value=2, key="f2")
        monetary2 = st.number_input("Total amount spent (R$)", min_value=0.0, value=300.0, key="m2")
        review = st.slider("Average review score given", 1.0, 5.0, 4.0)
        delivery_days = st.number_input("Average delivery days", min_value=0.0, value=10.0)
        on_time_rate = st.slider("On-time delivery rate", 0.0, 1.0, 0.9)
        installments = st.number_input("Average payment installments", min_value=1.0, value=2.0)
        run_churn = st.button("Predict churn risk", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if run_churn:
            with st.spinner(random.choice(CHURN_SPINNER_MSGS)):
                time.sleep(0.4)
                result = engine.predict_churn_risk(
                    frequency2, monetary2, review, delivery_days, on_time_rate, installments
                )
            kind = risk_kind(result["risk_label"])

            st.markdown('<div class="section-card" style="--accent:%s">' % COLOR_CHURN,
                         unsafe_allow_html=True)
            gc, tc = st.columns([1, 1])
            with gc:
                gauge_slot = st.empty()
                animate_gauge(gauge_slot, result["churn_probability"], COLOR_CHURN)
            with tc:
                st.markdown(f"#### {badge(result['risk_label'] + ' risk', kind)}", unsafe_allow_html=True)
                st.caption("Probability of no purchase in the next 180 days")

            st.markdown("**What's driving this model overall**")
            importances = engine.get_churn_feature_importance()
            imp_df = pd.DataFrame(
                [(FEATURE_LABELS.get(f, f), v) for f, v in importances],
                columns=["Feature", "Importance"],
            ).set_index("Feature")
            st.bar_chart(imp_df, color=COLOR_CHURN, horizontal=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.toast(f"Churn risk: {result['risk_label']}", icon="⚠️" if kind == "red" else "✅")

            st.session_state.history.append({
                "Model": "Churn risk",
                "Input": f"F={frequency2}, M={monetary2}, review={review}",
                "Result": f"{result['churn_probability']*100:.1f}% ({result['risk_label']})",
            })
        else:
            st.info("Fill in the profile and click **Predict churn risk** to see results here.")

# ----------------------------------------------------------------------------
# Tab 3 — Repeat purchase
# ----------------------------------------------------------------------------
with tab3:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown('<div class="section-card" style="--accent:%s">' % COLOR_REPEAT,
                     unsafe_allow_html=True)
        st.subheader("First order details")
        st.caption("Uses only what's known right after checkout - no purchase history required")
        price = st.number_input("Order price (R$)", min_value=0.0, value=120.0)
        freight = st.number_input("Freight value (R$)", min_value=0.0, value=18.0)
        installments3 = st.number_input("Payment installments", min_value=1, value=3, key="i3")
        payment_type = st.selectbox("Payment type", ["credit_card", "boleto", "voucher", "debit_card"])
        category = st.text_input("Product category", value="bed_bath_table")
        state = st.text_input("Customer state (2-letter)", value="SP")
        on_time = st.checkbox("Delivered on time", value=True)
        run_repeat = st.button("Predict repeat purchase", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if run_repeat:
            with st.spinner(random.choice(REPEAT_SPINNER_MSGS)):
                time.sleep(0.4)
                result = engine.predict_repeat_purchase(
                    price, freight, installments3, payment_type, category, state, on_time
                )
            will_return = result["repeat_purchase_probability"] >= 0.5
            kind = "green" if will_return else "amber"

            st.markdown('<div class="section-card" style="--accent:%s">' % COLOR_REPEAT,
                         unsafe_allow_html=True)
            gc, tc = st.columns([1, 1])
            with gc:
                gauge_slot = st.empty()
                animate_gauge(gauge_slot, result["repeat_purchase_probability"], COLOR_REPEAT)
            with tc:
                st.markdown(f"#### {badge(result['label'], kind)}", unsafe_allow_html=True)
                st.caption("Probability this first-time buyer places a second order")
            st.markdown('</div>', unsafe_allow_html=True)

            st.toast(result["label"], icon="🔁" if will_return else "🤔")
            if will_return:
                st.balloons()

            st.session_state.history.append({
                "Model": "Repeat purchase",
                "Input": f"{category}, R${price}, {payment_type}",
                "Result": f"{result['repeat_purchase_probability']*100:.1f}% ({result['label']})",
            })
        else:
            st.info("Fill in the order details and click **Predict repeat purchase** to see results here.")

# ----------------------------------------------------------------------------
# Tab 4 — History
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Predictions made this session")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            st.download_button(
                "Download CSV",
                hist_df.to_csv(index=False).encode("utf-8"),
                file_name="olist_predictions.csv",
                mime="text/csv",
            )
        with col2:
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("Nothing yet — predictions from the other tabs will show up here.")

st.divider()
st.caption("Team 3 · Customer Segmentation and Purchase Behavior Analysis · Olist dataset")