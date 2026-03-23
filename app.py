import streamlit as st
import pandas as pd
import joblib
import os

# ---------------- LOAD FILES ----------------
model = joblib.load("electricity_theft_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("Enter key meter details to detect possible electricity theft.")

# ---------------- IMPORTANT FEATURES (ONLY THESE SHOWN) ----------------
important_features = [
    "mtr_coef",
    "R_1",
    "usage_aux",
    "mtr_val_old",
    "mtr_code",
    "usage_1"
]

st.subheader("📥 Enter Key Meter Details")

user_input = {}

for col in important_features:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# ---------------- BUILD FULL INPUT ----------------
full_input = {}

for col in feature_columns:
    if col in user_input:
        full_input[col] = user_input[col]
    else:
        full_input[col] = 0  # default for missing features

input_df = pd.DataFrame([full_input])
input_df = input_df[feature_columns]

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
        return "High anomalies detected in electricity usage and meter behavior, indicating strong likelihood of theft."
    elif prob > 0.4:
        return "Moderate irregularities observed in usage patterns. Further inspection may be required."
    else:
        return "Electricity usage appears normal with no significant suspicious activity."

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    prob = model.predict_proba(input_df)[:, 1][0]
    pred = 1 if prob > threshold else 0
    risk = get_risk(prob)
    explanation = generate_explanation(prob)

    st.subheader("📊 Results")

    if pred == 1:
        st.error("⚠️ Theft Detected")
    else:
        st.success("✅ Normal Usage")

    st.write(f"**Probability of Theft:** {prob:.3f}")
    st.write(f"**Risk Level:** {risk}")

    st.subheader("🤖 AI Explanation")
    st.write(explanation)

# ---------------- RENDER SUPPORT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")
