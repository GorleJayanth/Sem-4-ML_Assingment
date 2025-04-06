import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from lime.lime_tabular import LimeTabularExplainer

def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values
    return X, y, list(df.drop(columns=["Label"]).columns)

def create_pipeline(model_type='classification'):
    base_models = {
        'classification': [
            ('dt', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier()),
            ('svm', SVC(probability=True))
        ],
        'regression': [
            ('dt', DecisionTreeRegressor()),
            ('knn', KNeighborsRegressor()),
            ('svm', SVR())
        ]
    }

    meta_models = {
        'classification': LogisticRegression(),
        'regression': Ridge()
    }

    stacking_model = StackingClassifier(estimators=base_models['classification'], final_estimator=meta_models['classification']) \
        if model_type == 'classification' else \
        StackingRegressor(estimators=base_models['regression'], final_estimator=meta_models['regression'])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacking', stacking_model)
    ])

    return pipeline
