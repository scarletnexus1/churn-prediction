# 📊 Customer Churn Prediction System

## 🚀 Overview

This project is an end-to-end machine learning system designed to predict customer churn in a telecom dataset. It goes beyond basic modeling by incorporating feature engineering, pipeline automation, API deployment, and an interactive UI.

---

## 🎯 Problem Statement

Customer churn is a critical business problem where companies lose customers over time. Predicting churn enables proactive retention strategies, reducing revenue loss.

---

## 🧠 Approach

### 1. Data Processing

* Cleaned missing values and corrected data types
* Used logical imputation (e.g., zero charges for new customers)

### 2. Feature Engineering (Core Focus)

* Behavioral features:

  * Customer engagement
  * Loyalty metrics
* Financial features:

  * Customer Lifetime Value (CLV)
  * Price sensitivity
* Interaction features:

  * Service usage vs tenure
* Non-linear transformations:

  * Log scaling for skewed data

### 3. Pipeline Design

* Built an end-to-end pipeline using `scikit-learn`
* Included:

  * Feature engineering
  * Imputation
  * Encoding (OneHotEncoder)
  * Scaling
  * Model training

---

## ⚙️ System Architecture

```text
User → Streamlit UI → FastAPI → ML Pipeline → Prediction
```

---

## 🧩 Components

### 🔹 Model (Pipeline)

* Handles preprocessing + prediction automatically
* Ensures consistency between training and inference

### 🔹 FastAPI Backend

* Serves predictions via REST API
* Decouples model from UI

### 🔹 Streamlit Frontend

* Interactive user interface
* Allows real-time predictions

---

## 📈 Model Performance

* Evaluated using:

  * Accuracy
  * Classification report
* Pipeline ensures consistent performance across environments

---

## 🔄 MLOps Concepts Applied

* Pipeline-based preprocessing
* Model versioning (extendable)
* Logging and retraining strategy (extendable)
* Separation of concerns (model/API/UI)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* FastAPI
* Streamlit
* Joblib

---

## ▶️ How to Run

### 1. Train Model

```bash
cd model
python train.py
```

### 2. Start API

```bash
cd api
uvicorn main:app --reload
```

### 3. Run UI

```bash
cd app
streamlit run app.py
```

---

## 📌 Key Learnings

* Importance of feature engineering over model complexity
* Building reproducible ML pipelines
* Designing scalable ML systems using APIs
* Understanding the ML lifecycle (EDA → deployment → monitoring)

---

## 🔮 Future Improvements

* Add SHAP-based explainability
* Implement automated retraining
* Add model monitoring (data drift detection)
* Deploy full system on cloud (Docker + CI/CD)

---

## 👨‍💻 Author

Nimit Arora
