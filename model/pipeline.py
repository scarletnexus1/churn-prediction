import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        services = [
            'PhoneService','MultipleLines','InternetService','OnlineSecurity',
            'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
        ]

        X['TotalServices'] = (X[services] == 'Yes').sum(axis=1)

        X['IsNewCustomer'] = (X['tenure'] == 0).astype(int)
        X['AvgCharges'] = X['TotalCharges'] / (X['tenure'] + 1)

        X['CLV'] = X['MonthlyCharges'] * X['tenure']
        X['ChargeConsistency'] = X['TotalCharges'] / (X['MonthlyCharges'] + 1)

        X['Engagement'] = X['tenure']
        X['ServiceDensity'] = X['TotalServices'] / (X['tenure'] + 1)

        X['PriceSensitivity'] = X['MonthlyCharges'] / (X['TotalServices'] + 1)
        X['LoyaltyValue'] = X['tenure'] * X['TotalServices']

        X['LogCharges'] = np.log1p(X['TotalCharges'])

        return X