import streamlit as st
import pandas as pd
import joblib
import os
import requests

# ---------------- LOAD MODEL ----------------
model = joblib.load("electricity_theft_model.pkl")

# ✅ EXACT training features (DO NOT CHANGE)
feature_columns = [
    "mtr_tariff","mtr_status","mtr_code","mtr_notes","mtr_coef",
    "usage_1","usage_2","usage_3","usage_4",
    "mtr_val_old","mtr_val_new","months_num","mtr_type",
    "usage_aux","usage_n_aux","date_flip_flag",
    "date_overlap_invoice","date_overlap_months",
    "months_num_calc",
    "R_1","R_2a","R_2b","R_3a","R_3b",
    "idx_prv","idx_nxt"
]

# ---------------- DEFAULT VALUES ----------------
sample_dict = {
    "mtr_tariff": 40,
    "mtr_status": 0,
    "mtr_code": 5,
    "mtr_notes": 8,
    "mtr_coef": 1,
    "usage_1": 28,
    "usage_2": 0,
    "usage_3": 0,
    "usage_4": 0,
    "mtr_val_old": 5125,
    "mtr_val_new": 5125,
    "months_num": 2,
    "mtr_type": 1,
    "usage_aux": 0,
    "usage_n_aux": 28,
    "date_flip_flag": 1,
    "date_overlap_invoice": 0,
    "date_overlap_months": 0,
    "months_num_calc": 7.8,
    "R_1": -1,
    "R_2a": -1,
    "R_2b": 0,
    "R_3a": -1,
    "R_3b": 0,
    "idx_prv": 42732,
    "idx_nxt": 42734
}

# ---------------- UI ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("Enter meter details to detect possible electricity theft.")

# ---------------- INPUT ----------------
important_features = [
    "mtr_coef","usage_1","usage_2","usage_3","usage_4",
    "mtr_val_old","mtr_val_new","months_num"
]

st.subheader("📥 Enter Meter Details")

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

# ---------------- RULE-BASED EXPLANATION ----------------
def generate_rule_explanation(prob, input_df):
    row = input_df.iloc[0]

    usage_total = row["usage_1"] + row["usage_2"] + row["usage_3"] + row["usage_4"]
    meter_diff = row["mtr_val_new"] - row["mtr_val_old"]

    text = ""

    if prob > 0.8:
        text += "⚠️ High risk detected.\n\n"
    elif prob > 0.4:
        text += "⚠️ Moderate irregularities.\n\n"
    else:
        text += "✅ Usage appears normal.\n\n"

    text += f"- Total Consumption: {usage_total}\n"
    text += f"- Meter Difference: {meter_diff}\n"
    text += f"- Billing Months: {row['months_num']}\n\n"

    if meter_diff == 0:
        text += "• No meter change observed.\n"
    elif meter_diff > usage_total * 5:
        text += "• Meter increased unusually compared to usage.\n"
    else:
        text += "• Meter readings are consistent.\n"

    return text

# ---------------- AI EXPLANATION (UPGRADED) ----------------
def generate_explanation_llm(prob, input_df):

    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    HF_TOKEN = os.getenv("HF_API_KEY")

    if not HF_TOKEN:
        return "⚠️ AI explanation unavailable (API key missing)."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    row = input_df.iloc[0]

    usage_total = row["usage_1"] + row["usage_2"] + row["usage_3"] + row["usage_4"]
    meter_diff = row["mtr_val_new"] - row["mtr_val_old"]

    prompt = f"""
    You are an electricity fraud detection expert.

    Analyze the data below and explain clearly:

    Theft Probability: {prob:.2f}

    Data:
    - Total Consumption: {usage_total}
    - Meter Difference: {meter_diff}
    - Billing Months: {row['months_num']}

    Explain:
    1. Is this normal or suspicious?
    2. Why?
    3. Keep it simple and professional.

    Answer:
    """

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=10
        )

        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]

    except:
        pass

    return "AI explanation not available."

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    prob = model.predict_proba(input_df)[:, 1][0]

    threshold = 0.6
    pred = 1 if prob > threshold else 0

    risk = get_risk(prob)

    rule_text = generate_rule_explanation(prob, input_df)
    ai_text = generate_explanation_llm(prob, input_df)

    st.subheader("📊 Results")

    if pred == 1:
        st.error("⚠️ Theft Detected")
    else:
        st.success("✅ Normal Usage")

    st.write(f"**Probability of Theft:** {prob:.2f}")
    st.write(f"**Risk Level:** {risk}")

    st.subheader("📊 System Explanation")
    st.info(rule_text)

    st.subheader("🤖 AI Insight")
    st.success(ai_text)

# ---------------- RENDER SUPPORT ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")
