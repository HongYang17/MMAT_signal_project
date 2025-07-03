# === volatility/vol_model.py ===

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

class VolatilityModel:
    def __init__(self, df):
        self.df = df.copy()
        self.pipeline = None
        self.selected_features = None

    def label_future_volatility(self, future_window=45, threshold=0.003):
        self.df['future_high'] = self.df['high'].rolling(future_window, min_periods=1).max().shift(-future_window)
        self.df['future_low'] = self.df['low'].rolling(future_window, min_periods=1).min().shift(-future_window)
        self.df['future_range'] = (self.df['future_high'] - self.df['future_low']) / self.df['close']
        self.df['vol_target'] = (self.df['future_range'].abs() > threshold).astype(int)
        return self.df

    def prepare_data(self):
        drop_cols = ['target', 'next_close', 'return', 'future_high', 'future_low', 'future_range']
        feature_cols = [col for col in self.df.columns if col not in drop_cols + ['vol_target']]
        X = self.df[feature_cols]
        y = self.df['vol_target']
        valid = y.notna() & X.notna().all(axis=1)
        return X[valid], y[valid]

    def train(self, importance_threshold=0.01):
        X, y = self.prepare_data()
        tscv = TimeSeriesSplit(n_splits=3)
        best_score = 0

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
                max_depth=4,
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
                self.pipeline = pipeline
                importances = model.feature_importances_
                selected_idx = np.where(importances >= importance_threshold)[0]
                self.selected_features = X.columns[selected_idx].tolist()

        return self.pipeline, self.selected_features

    def predict(self, df_latest):
        X_latest = df_latest[self.selected_features].copy()
        if X_latest.isnull().any().any():
            raise ValueError("Volatility input contains NaNs.")
        return self.pipeline.predict(X_latest)[0], self.pipeline.predict_proba(X_latest)[0][1]

    def save(self, model_path='models/volatility_model.pkl', feature_path='models/volatility_features.pkl'):
        joblib.dump(self.pipeline, model_path)
        joblib.dump(self.selected_features, feature_path)

    def load(self, model_path='models/volatility_model.pkl', feature_path='models/volatility_features.pkl'):
        self.pipeline = joblib.load(model_path)
        self.selected_features = joblib.load(feature_path)
        return self
