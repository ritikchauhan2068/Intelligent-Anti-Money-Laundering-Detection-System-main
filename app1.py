import streamlit as st
import pandas as pd
import joblib
from pathlib import Path



st.set_page_config(
    page_title="AML Detection System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Intelligent Anti-Money Laundering Detection System")

st.write(
    "Enter transaction details below to identify whether "
    "a financial transaction is legitimate or potentially suspicious."
)

st.divider()



BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "AML_best_model.pkl")

try:
    model = load_model()
    model_name = type(model.named_steps["Model"]).__name__
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()



currency_options = [
    'UK pounds', 'Indian rupee', 'Albanian lek',
    'Swiss franc', 'Pakistani rupee', 'Naira',
    'Yen', 'Euro', 'Dirham', 'Mexican Peso',
    'Turkish lira', 'US dollar', 'Moroccan dirham'
]

country_options = [
    'UK', 'Albania', 'Nigeria', 'Japan', 'Spain',
    'Switzerland', 'UAE', 'Italy', 'France',
    'Pakistan', 'Mexico', 'Austria', 'Germany',
    'Turkey', 'India', 'Morocco', 'USA', 'Netherlands'
]

payment_options = [
    'Cash Deposit', 'Cross-border', 'Cheque',
    'ACH', 'Credit card', 'Debit card',
    'Cash Withdrawal'
]



with st.form("transaction_form"):

    st.subheader("Transaction Details")

    AMOUNT = st.number_input(
        "Transaction Amount",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )

    col1, col2 = st.columns(2)

    with col1:
        Payment_Currency = st.selectbox("Payment Currency", currency_options)
        Sender_bank_location = st.selectbox("Sender Bank Location", country_options)
        Payment_type = st.selectbox("Payment Type", payment_options)

    with col2:
        Received_Currency = st.selectbox("Received Currency", currency_options)
        Receiver_bank_location = st.selectbox("Receiver Bank Location", country_options)

    st.divider()

    submit = st.form_submit_button(
        "🔍 Analyze Transaction",
        use_container_width=True
    )



if submit:

    # Same column names and order as used in training
    input_data = pd.DataFrame({
        'Amount': [AMOUNT],
        'Payment_currency': [Payment_Currency],
        'Received_currency': [Received_Currency],
        'Sender_bank_location': [Sender_bank_location],
        'Receiver_bank_location': [Receiver_bank_location],
        'Payment_type': [Payment_type]
    })

    st.subheader("Transaction Summary")
    st.dataframe(input_data, use_container_width=True)

    try:
        # The pipeline does the preprocessing itself.
        # SMOTE is skipped automatically at prediction time.
        probability = float(model.predict_proba(input_data)[0][1])

        if probability >= 0.5:
            st.error("🚨 Potentially Suspicious Transaction Detected!")
            st.write(
                f"The {model_name} model has classified this transaction "
                "as potentially suspicious."
            )
        else:
            st.success("✅ Legitimate Transaction")
            st.write(
                f"The {model_name} model has classified this transaction "
                "as legitimate."
            )

        st.metric(
            "Suspicious Transaction Probability",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "Note: This is a machine learning prediction, not a final "
            "determination of money laundering. Suspicious "
            "transactions should undergo further investigation."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------------
# Footer
# -----------------------------------

st.divider()

st.caption(
    "Intelligent Anti-Money Laundering Detection System | "
    f"Powered by Machine Learning ({model_name})"
)
