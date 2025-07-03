# === signals/model.py ===

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

class SignalModel:
    def __init__(self, df, target_col='target'):
        self.df = df.copy()
        self.target_col = target_col
        self.pipeline = None
        self.selected_features = None

    def prepare_data(self, features):
        X = self.df[features]
        y = self.df[self.target_col]
        valid = y.notna() & X.notna().all(axis=1)
        return X[valid], y[valid]

    def train(self, importance_threshold=0.01, n_splits=3):
        X, y = self.prepare_data([col for col in self.df.columns if col != self.target_col])
        tscv = TimeSeriesSplit(n_splits=n_splits)

        best_score = 0
        best_pipeline = None
        selected_features = None

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, X.columns.tolist())
            ])

            scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc'
            )

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)

            if auc > best_score:
                best_score = auc
                best_pipeline = pipeline
                importances = model.feature_importances_
                selected_idx = np.where(importances >= importance_threshold)[0]
                selected_features = X.columns[selected_idx].tolist()

        self.pipeline = best_pipeline
        self.selected_features = selected_features
        return self.pipeline, self.selected_features

    def predict(self, df_new):
        if self.pipeline is None or self.selected_features is None:
            raise ValueError("Model has not been trained or loaded.")

        X = df_new[self.selected_features].copy()
        if X.isnull().any().any():
            raise ValueError("Input data contains NaNs in selected features.")

        return self.pipeline.predict(X), self.pipeline.predict_proba(X)[:, 1]

    def save(self, model_path='models/final_signal_model.pkl', feature_path='models/selected_features.pkl'):
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.selected_features, feature_path)

    def load(self, model_path='models/final_signal_model.pkl', feature_path='models/selected_features.pkl'):
        self.pipeline = joblib.load(model_path)
        self.selected_features = joblib.load(feature_path)
        return self
