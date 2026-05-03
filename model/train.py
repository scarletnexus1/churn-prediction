import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
import sys
import os



# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from app.pipeline import FeatureEngineer
# Load data
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.loc[df['tenure'] == 0, 'TotalCharges'] = 0

# Drop ID
df.drop('customerID', axis=1, inplace=True)

# Target
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
scale_pos_weight = class_weights[1] / class_weights[0]

# Column types
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Full pipeline
model_pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('preprocessing', preprocessor),
    ('model', XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Train
model_pipeline.fit(X_train, y_train)

# Predictions
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\n Model Performance")

print(f"Accuracy     : {acc:.4f}")
print(f"ROC-AUC      : {roc:.4f}")
print(f"Precision    : {precision_score(y_test, y_pred):.4f}")
print(f"Recall       : {recall_score(y_test, y_pred):.4f}")  # 🔥 IMPORTANT
print(f"F1 Score     : {f1_score(y_test, y_pred):.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
# Save
joblib.dump(model_pipeline, "pipeline_model.pkl")

print("✅ Model saved successfully!")