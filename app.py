import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD FILES ----------------
model = joblib.load("electricity_theft_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("Enter the meter details to detect possible electricity theft.")

# ---------------- USER INPUT ----------------
st.subheader("📥 Input Data")

user_input = {}

for col in feature_columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

input_df = pd.DataFrame([user_input])
input_df = input_df[feature_columns]  # maintain order

# ---------------- LOGIC ----------------
threshold = 0.2

def get_risk(prob):
    if prob > 0.8:
        return "🔴 HIGH RISK"
    elif prob > 0.4:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

def generate_explanation(prob):
    if prob > 0.8:
        return "The model detected strong anomalies in usage patterns and meter behavior, indicating a high likelihood of electricity theft."
    elif prob > 0.4:
        return "Some irregularities were observed in electricity usage, suggesting a moderate risk of theft."
    else:
        return "The electricity usage pattern appears normal with no significant anomalies detected."

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    prob = model.predict_proba(input_df)[:, 1][0]
    pred = 1 if prob > threshold else 0
    risk = get_risk(prob)
    explanation = generate_explanation(prob)

    st.subheader("📊 Results")

    # Prediction
    if pred == 1:
        st.error("⚠️ Theft Detected")
    else:
        st.success("✅ Normal Usage")

    # Probability
    st.write(f"**Probability of Theft:** {prob:.3f}")

    # Risk
    st.write(f"**Risk Level:** {risk}")

    # Explanation
    st.subheader("🤖 AI Explanation")
    st.write(explanation)
