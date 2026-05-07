import streamlit as st
import pandas as pd
import joblib
import shap
import os
import sys
import pickle
import numpy as np

import traceback

# Fixed paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from pipeline import FeatureEngineer

# Custom unpickler to force correct module resolution
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'FeatureEngineer':
            return FeatureEngineer
        return super().find_class(module, name)

model_path = os.path.join(BASE_DIR, "..", "model", "pipeline_model.pkl")

with open(model_path, 'rb') as f:
    model = CustomUnpickler(f).load()

st.set_page_config(page_title="Churn Predictor", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("📥 Enter Customer Details")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.slider("Monthly Charges", 0, 150, 70)
total = st.slider("Total Charges", 0, 10000, 1000)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# CREATE INPUT
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract": contract,
    "PaymentMethod": payment
}

# PREDICT
if st.button("🚀 Predict Churn"):


    expected_cols = list(model.named_steps['preprocessing'].feature_names_in_)


    base_data = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': int(tenure),
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': contract,
        'PaperlessBilling': 'Yes',
        'PaymentMethod': payment,
        'MonthlyCharges': float(monthly),
        'TotalCharges': float(total)
    }

    # creating df with expected cols
    input_df = pd.DataFrame(columns=expected_cols)

    # fill known value
    for col in base_data:
        if col in input_df.columns:
            input_df.loc[0, col] = base_data[col]

    # Fill missing safely
    for col in input_df.columns:
        if input_df[col].isnull().all():
            input_df.loc[0, col] = 'No' if input_df[col].dtype == 'object' else 0

    #  ALIGNING NUMERIC TYPES
    input_df['tenure'] = input_df['tenure'].astype(int)
    input_df['MonthlyCharges'] = input_df['MonthlyCharges'].astype(float)
    input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)

    #  MAKING clean index
    input_df = input_df.reset_index(drop=True)

    # Debugging step (you can remove later)
    st.write("INPUT:", input_df)

    # Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # RESULT
    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn ({prob:.2f})")
    else:
        st.success(f"✅ Customer Likely to Stay ({1-prob:.2f})")

    # RISK SCORE

    st.subheader("🎯 Churn Risk Score")

    st.metric("Churn Probability", f"{prob*100:.1f}%")
    st.progress(int(prob * 100))

    if prob > 0.7:
        st.error("🔴 High Risk Customer")
    elif prob > 0.4:
        st.warning("🟠 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")

    # CUSTOMER INSIGHTS
    st.subheader("📊 Customer Insights")

    avg_charges = total / (tenure + 1)
    clv = monthly * tenure
    price_sensitivity = monthly / (tenure + 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Avg Charges", f"{avg_charges:.2f}")
    col2.metric("📈 CLV", f"{clv:.2f}")
    col3.metric("⚡ Price Sensitivity", f"{price_sensitivity:.2f}")

    # REASON FOR THIS PREDICTION

    st.subheader("🧠 Why this prediction?")

    if prediction == 1:
        st.write("This customer shows patterns similar to users who have churned in the past.")
    else:
        st.write("This customer shows stable behavior similar to users who stayed.")

    # SHAP EXPLANATION
    # SHAP EXPLANATION
    try:
        fe = model.named_steps['feature_engineering']
        X_fe = fe.transform(input_df)

        preprocessor = model.named_steps['preprocessing']
        X_processed = preprocessor.transform(X_fe)

        X_processed = X_processed.toarray() if hasattr(X_processed, 'toarray') else np.array(X_processed,
                                                                                             dtype=np.float64)

        feature_names = list(preprocessor.get_feature_names_out())
        final_model = model.named_steps['model']

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_processed)

        # Random Forest binary — shap_values is a list of 2 arrays
        # Index [1] = churn class, [0] = first customer
        sv = shap_values[0]

        st.subheader("🔍 Key Drivers of Prediction")

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": sv
        })

        shap_df["AbsImpact"] = shap_df["Impact"].abs()
        top_features = shap_df.sort_values(by="AbsImpact", ascending=False).head(5)

        top_features["Feature"] = top_features["Feature"] \
            .str.replace("cat__", "") \
            .str.replace("num__", "")


        for _, row in top_features.iterrows():

            feature_name = row['Feature'] \
                .replace("cat__", "") \
                .replace("num__", "") \
                .replace("_", " ")

            feature_name = feature_name.title()
            if row["Impact"] > 0:
                st.markdown(
                    f"""
                    <div style='padding:0.7rem 1rem; margin:0.4rem 0; border-radius:8px;
                    background:rgba(224,85,85,0.08); border-left:4px solid #E05555;'>
                    🔺 <strong>{feature_name}</strong> increases churn risk
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f"""
                    <div style='padding:0.7rem 1rem; margin:0.4rem 0; border-radius:8px;
                    background:rgba(76,175,130,0.08); border-left:4px solid #4CAF82;'>
                    🔻 <strong>{feature_name}</strong> decreases churn risk
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"SHAP Error: {e}")
        st.code(traceback.format_exc())

    # RECOMMENDATIONS
    st.subheader("💡 Recommendations")


    recommendations = []

    if tenure < 12:
        recommendations.append("Improve onboarding experience")

    if contract == "Month-to-month":
        recommendations.append("Offer long-term contract discount")

    if monthly > 80:
        recommendations.append("Consider pricing optimization")

    if len(recommendations) == 0:
        recommendations.append("Customer looks stable")

    for r in recommendations:
        st.markdown(f"👉 {r}")

# FOOTER
st.markdown("---")
st.markdown("🚀 Built by NIMIT ARORA | ML Project")