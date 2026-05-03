🚀 Customer Churn Prediction Dashboard

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn, along with actionable business insights and recommendations.

📌 Project Overview

This project builds a customer churn prediction system using machine learning and deploys it as an interactive web application using Streamlit Cloud.

The app allows users to input customer details and get:

✅ Churn prediction (Stay / Churn)
📊 Churn probability score
📈 Customer insights (CLV, pricing sensitivity)
🔍 Key drivers influencing prediction
💡 Business recommendations
❗ Problem Statement

Customer churn is a major challenge in the telecom industry.

Acquiring a new customer costs 5x more than retaining an existing one.

👉 The goal of this project is to:

Predict whether a customer will churn
Identify key factors influencing churn
Provide actionable strategies to reduce churn

🛠️ Tech Stack
Python
Pandas, NumPy → Data processing
Scikit-learn → Model building (Pipeline, Preprocessing)
Matplotlib, SHAP → Visualization & explainability
Streamlit → Web app UI
Joblib → Model serialization
Git & GitHub → Version control
Streamlit Cloud → Deployment

🤖 Model Performance
Metric	Value
Accuracy	81%
Precision (Churn)	0.64
Recall (Churn)	0.57
F1 Score	0.60

📊 Interpretation:
Model performs well overall with 81% accuracy
Balanced performance between precision and recall
Effectively identifies churn-prone customers

📸 Application Screenshots

🖥️ Dashboard Interface
![Dashboard](assets/app1.png)

📊 Prediction Output
![Prediction](app2/dashboard.png)

📈 Insights & Recommendations
![Insights](assets/app3.png)

🌐 Live Demo

👉 Try the app here:
🔗 https://churn-prediction-scarlet.streamlit.app/

(Replace with your actual deployed link)

📊 Key Features
🎯 Real-time churn prediction
📉 Risk scoring system
📊 Customer insights (CLV, pricing sensitivity)
🔍 Explainable AI (feature drivers)
💡 Smart recommendations
🧠 Business Insights
🔴 1. Short Tenure = High Risk
Customers with low tenure (e.g., ~3 months) are more likely to churn

👉 Reason:

New users are still evaluating the service
Low commitment level
🟡 2. High Monthly Charges Increase Churn Risk
Customers paying higher monthly charges are more price-sensitive
🟢 3. Long-Term Customers Are Stable
High TotalCharges → Long tenure → Low churn probability
💡 Business Recommendations
✅ 1. Promote Long-Term Contracts
Customers with short tenure should be encouraged to switch to long-term plans

✔ Offer incentives like:

Discounted yearly plans
Bundled services
✅ 2. Offer Discounts to At-Risk Customers
Provide targeted discounts to customers likely to churn

✔ Example:

“Switch to yearly plan and get 20% off”
✅ 3. Improve Early Customer Experience
Focus on onboarding for new customers (first 3–6 months)

✔ Provide:

Better support
Tutorials / guidance
Personalized engagement
🧠 Key Learning
The biggest challenge in ML deployment is not model building,
but ensuring consistent data schema and environment compatibility.
👨‍💻 Author

Nimit Arora
🚀 Machine Learning Enthusiast

⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!

🔥 If you want next upgrade

I can help you:

Improve model (XGBoost / better recall)
Add advanced SHAP visualizations
Make UI premium (dashboard-level)

Just tell me 🚀