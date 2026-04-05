import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# ---------------- LOAD MODEL ----------------
model = joblib.load("electricity_theft_model.pkl")

# ---------------- UI ----------------
st.set_page_config(page_title="Electricity Theft Detection", layout="centered")

st.title("⚡ Electricity Theft Detection System")
st.write("Enter meter details to assess risk of electricity theft.")

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
    "idx_nxt": 42734,
    "idx": 1000,
    "year": 2020,
    "month": 6
}

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

# Match training features
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# ---------------- RISK FUNCTION ----------------
def get_risk(prob_theft):
    if prob_theft > 0.7:
        return "🔴 HIGH RISK"
    elif prob_theft > 0.4:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"

# ---------------- GRAPH ----------------
def plot_probabilities(prob_theft, prob_normal):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Theft", "Normal"],
        y=[prob_theft, prob_normal],
        text=[f"{prob_theft:.2f}", f"{prob_normal:.2f}"],
        textposition='auto'
    ))

    fig.update_layout(title="Prediction Probabilities")
    return fig

# ---------------- FEATURE IMPORTANCE PER PREDICTION ----------------
def explain_prediction(input_df):
    importances = model.feature_importances_
    features = model.feature_names_in_

    values = input_df.iloc[0]

    # Simple impact score = value * importance
    impact = values * importances

    df_imp = pd.DataFrame({
        "Feature": features,
        "Impact": impact
    })

    df_imp = df_imp.sort_values(by="Impact", ascending=False)

    return df_imp.head(5)

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze"):

    prob_normal = model.predict_proba(input_df)[:, 1][0]
    prob_theft = 1 - prob_normal

    risk = get_risk(prob_theft)

    st.subheader("📊 Results")

    if prob_theft > 0.7:
        st.error(f"🔴 High Risk of Theft ({prob_theft:.2f})")
    elif prob_theft > 0.4:
        st.warning(f"🟡 Medium Risk ({prob_theft:.2f})")
    else:
        st.success(f"🟢 Low Risk ({prob_theft:.2f})")

    st.write(f"🔴 **Theft Probability:** {prob_theft:.2f}")
    st.write(f"🟢 **Normal Probability:** {prob_normal:.2f}")
    st.write(f"**Risk Level:** {risk}")

    # 📈 Graph
    st.subheader("📈 Probability Visualization")
    st.plotly_chart(plot_probabilities(prob_theft, prob_normal), use_container_width=True)

    # 🔍 Feature Importance
    st.subheader("🔍 Top Influencing Features")
    imp_df = explain_prediction(input_df)
    st.dataframe(imp_df)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    os.system(f"streamlit run app.py --server.port {port} --server.address 0.0.0.0")
