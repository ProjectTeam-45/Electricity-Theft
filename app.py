import streamlit as st
import pandas as pd
import joblib
import os
import requests

# ---------------- LOAD MODEL ----------------
model = joblib.load("electricity_theft_model.pkl")

# ✅ Correct feature columns (MUST match training)
feature_columns = [
    "mtr_tariff","mtr_id","mtr_status","mtr_code","mtr_notes","mtr_coef",
    "usage_1","usage_2","usage_3","usage_4",
    "mtr_val_old","mtr_val_new","months_num","mtr_type",
    "usage_aux","usage_n_aux","date_flip_flag","date_overlap_invoice",
    "date_overlap_months","months_num_calc",
    "R_1","R_2a","R_2b","R_3a","R_3b",
    "idx","idx_prv","idx_nxt","year","month"
]

# ---------------- SAMPLE DEFAULT VALUES ----------------
sample_dict = {col: 0 for col in feature_columns}

sample_dict.update({
    "mtr_tariff": 11,
    "mtr_coef": 1,
    "usage_1": 200,
    "mtr_val_old": 14000,
    "mtr_val_new": 14200,
    "months_num": 4,
    "year": 2024,
    "month": 6
})

# ---------------- UI ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")

# ---------------- INPUT ----------------
important_features = [
    "mtr_coef",
    "usage_1",
    "mtr_val_old",
    "mtr_val_new",
    "months_num"
]

st.subheader("📥 Enter Key Meter Details")

user_input = {}

for col in important_features:
    user_input[col] = st.number_input(f"{col}", value=float(sample_dict[col]))

# ---------------- BUILD INPUT ----------------
full_input = sample_dict.copy()
full_input.update(user_input)

input_df = pd.DataFrame([full_input])
input_df = input_df[feature_columns]

# ---------------- RISK LOGIC ----------------
def get_risk(prob):
    if prob > 0.8:
        return "🔴 HIGH RISK"
    elif prob > 0.4:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# ---------------- LLM (FIXED) ----------------
def generate_explanation_llm(prob, input_df):

    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    
    HF_TOKEN = os.getenv("HF_API_KEY")  # ✅ correct way

    if not HF_TOKEN:
        return fallback_explanation(prob)

    headers = {"Authorization": f"Bearer {'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'}"}

    prompt = f"""
    Electricity theft detection system:

    Probability: {prob:.2f}
    Explain simply if this is theft or normal usage.
    """

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=10)
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]

    except:
        pass

    return fallback_explanation(prob)

# ---------------- FALLBACK ----------------
def fallback_explanation(prob):
    if prob > 0.8:
        return "High probability of electricity theft due to abnormal patterns."
    elif prob > 0.4:
        return "Moderate irregularities detected."
    else:
        return "Usage appears normal."

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    prob = model.predict_proba(input_df)[:, 1][0]

    # ✅ Keep threshold aligned with your model tuning
    threshold = 0.6
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
