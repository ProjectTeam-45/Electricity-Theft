import streamlit as st
import pandas as pd
import joblib
import os
import requests

# ---------------- LOAD FILES ----------------
model = joblib.load("electricity_theft_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ---------------- NORMAL SAMPLE (FROM NOTEBOOK) ----------------
sample_dict = {
    "mtr_tariff": 11.0,
    "mtr_status": 0.0,
    "mtr_code": 442.0,
    "mtr_notes": 9.0,
    "mtr_coef": 1.0,
    "usage_1": 200.0,
    "usage_2": 100.0,
    "usage_3": 200.0,
    "usage_4": 4354.0,
    "mtr_val_old": 437184.0,
    "mtr_val_new": 442038.0,
    "months_num": 1.0,
    "mtr_type": 0.0,
    "usage_aux": 0.0,
    "usage_n_aux": 4854.0,
    "date_flip_flag": 1.0,
    "date_overlap_invoice": 0.0,
    "date_overlap_months": 0.0,
    "months_num_calc": 1.1,
    "R_1": 0.0,
    "R_2a": 0.0,
    "R_2b": 0.0,
    "R_3a": 1.0,
    "R_3b": 0.0,
    "idx_prv": 60464.0,
    "idx_nxt": 60466.0
}

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("Enter key meter details to detect possible electricity theft.")

# ---------------- IMPORTANT FEATURES ----------------
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
    user_input[col] = st.number_input(f"{col}", value=sample_dict[col])

# ---------------- BUILD FULL INPUT ----------------
full_input = {}

for col in feature_columns:
    if col in user_input:
        full_input[col] = user_input[col]
    else:
        full_input[col] = sample_dict[col]  # 🔥 Use real sample instead of 0

input_df = pd.DataFrame([full_input])
input_df = input_df[feature_columns]

# ---------------- LOGIC ----------------
def get_risk(prob):
    if prob > 0.8:
        return "🔴 HIGH RISK"
    elif prob > 0.4:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# ---------------- HUGGING FACE LLM ----------------
def generate_explanation_llm(prob, input_df):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    
    headers = {"Authorization": f"Bearer {os.getenv('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')}"}

    prompt = f"""
    Electricity theft detection system:

    Probability: {prob:.2f}
    Data: {input_df.to_dict()}

    Explain in simple terms why this is theft or normal usage.
    """

    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return fallback_explanation(prob)

    except:
        return fallback_explanation(prob)

# ---------------- FALLBACK ----------------
def fallback_explanation(prob):
    if prob > 0.8:
        return "High probability of electricity theft due to abnormal consumption patterns."
    elif prob > 0.4:
        return "Moderate irregularities detected. Further inspection may be required."
    else:
        return "Electricity usage appears normal with no significant anomalies."

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    prob = model.predict_proba(input_df)[:, 1][0]

    threshold = 0.8
    pred = 1 if prob > threshold else 0

    risk = get_risk(prob)
    explanation = generate_explanation_llm(prob, input_df)

    st.subheader("📊 Results")

    if pred == 1:
        st.error("⚠️ Theft Detected")
    else:
        st.success("✅ Normal Usage")

    st.write(f"**Probability of Theft:** {prob:.2f}")
    st.write(f"**Risk Level:** {risk}")

    st.subheader("🤖 AI Explanation")
    st.info(explanation)

# ---------------- RENDER SUPPORT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")
