from fastapi import FastAPI
import joblib
import pandas as pd
import os
import sys
from pydantic import BaseModel

sys.path.append(os.path.abspath("../model"))

app = FastAPI()

# Fix path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "pipeline_model.pkl")

model = joblib.load(model_path)

# Input schema
class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerInput):

    data = data.dict()

    # Default features
    data['gender'] = 'Male'
    data['SeniorCitizen'] = 0
    data['Partner'] = 'No'
    data['Dependents'] = 'No'
    data['PaperlessBilling'] = 'Yes'

    data['PhoneService'] = 'Yes'
    data['MultipleLines'] = 'No'
    data['InternetService'] = 'DSL'
    data['OnlineSecurity'] = 'No'
    data['OnlineBackup'] = 'No'
    data['DeviceProtection'] = 'No'
    data['TechSupport'] = 'No'
    data['StreamingTV'] = 'No'
    data['StreamingMovies'] = 'No'

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(prob)
    }