import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import config
from signals.model_preparation import prepare_model_data

DEFAULT_MODEL_PATH = config.XGB_MODEL_FILE
DEFAULT_META_PATH = config.XGB_FEATURES_FILE


class SignalModel:
    def __init__(self):
        self.cv_pipeline = None
        self.final_model = None
        self.selected_features = None
        self.importances = {}
        self.last_trained = None

    def train_stage1(self, df, importance_threshold=0.01, n_splits=3):
        X, y, all_features = prepare_model_data(df)

        cutoff_date = X.index.max() - pd.Timedelta(days=30)
        X_old = X[X.index < cutoff_date]
        y_old = y[X.index < cutoff_date]

        best_score = 0
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_old)):
            X_train, X_test = X_old.iloc[train_idx], X_old.iloc[test_idx]
            y_train, y_test = y_old.iloc[train_idx], y_old.iloc[test_idx]

            preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), all_features)
            ])

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
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
            print(f"[Stage1][Fold {fold+1}] AUC: {auc:.4f}")

            if auc > best_score:
                best_score = auc
                self.cv_pipeline = pipeline
                importances = model.feature_importances_
                idx = np.where(importances >= importance_threshold)[0]
                self.selected_features = [all_features[i] for i in idx]
                self.importances = {all_features[i]: float(importances[i]) for i in idx}

        if not self.selected_features:
            raise RuntimeError("[Stage1] No features selected.")

        self.last_trained = datetime.utcnow()
        print(f"[Stage1] Selected {len(self.selected_features)} features.")
        return self  # ✅

    def retrain_stage2(self, df):
        X, y, _ = prepare_model_data(df)
        cutoff_date = X.index.max() - pd.Timedelta(days=30)
        X_recent = X[X.index >= cutoff_date]
        y_recent = y[X.index >= cutoff_date]

        valid_features = [f for f in self.selected_features if f in X_recent.columns]
        if not valid_features:
            raise ValueError("[Stage2] No selected features found in recent data.")

        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), valid_features)
        ])

        model = clone(self.cv_pipeline.named_steps['classifier'])

        self.final_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Extract X and y, keeping only valid features
        X_fit = X_recent[valid_features].copy()
        X_fit.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_fit.dropna(inplace=True)
        y_fit = y_recent.loc[X_fit.index]

        if X_fit.empty or y_fit.empty:
            raise ValueError("[Stage2] No clean data left after removing NaNs/Infs.")

        self.final_model.fit(X_fit, y_fit)
        print("[Stage2] Final model retrained successfully.")
        self.last_trained = datetime.utcnow()
        return self

    def predict(self, df_new):
        if self.final_model is None or self.selected_features is None:
            raise ValueError("[Predict] Final model not trained.")

        missing = set(self.selected_features) - set(df_new.columns)
        if missing:
            print(f"[Predict][Warning] Filling missing features: {missing}")
            for col in missing:
                df_new[col] = 0.0

        X = df_new[self.selected_features].copy()
        if X.isnull().any().any():
            raise ValueError(f"[Predict] NaNs found in: {X.columns[X.isnull().any()].tolist()}")

        preds = self.final_model.predict(X)
        probs = self.final_model.predict_proba(X)[:, 1]
        return preds, probs

    def save(self, model_path=DEFAULT_MODEL_PATH, feature_path=DEFAULT_META_PATH):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)

        joblib.dump(self.final_model, model_path)
        metadata = {
            "features": self.selected_features,
            "importances": self.importances,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None
        }
        with open(feature_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Save] Model saved to {model_path}")
        print(f"[Save] Features saved to {feature_path}")

    def load(self, model_path=DEFAULT_MODEL_PATH, feature_path=DEFAULT_META_PATH):
        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            raise FileNotFoundError("[Load] Missing model or metadata.")

        self.final_model = joblib.load(model_path)
        with open(feature_path, "r") as f:
            meta = json.load(f)
            self.selected_features = meta["features"]
            self.importances = meta.get("importances", {})
            last_trained_str = meta.get("last_trained", None)
            self.last_trained = datetime.fromisoformat(last_trained_str) if last_trained_str else None

        print("[Load] Model and features loaded.")
        if self.last_trained:
            print(f"[Load] Last trained: {self.last_trained.isoformat()}")
        return self

    def train_and_save(self, df, importance_threshold=0.01, n_splits=3,
                       model_path=DEFAULT_MODEL_PATH, feature_path=DEFAULT_META_PATH):
        return self.train_stage1(df, importance_threshold, n_splits) \
                   .retrain_stage2(df) \
                   .save(model_path, feature_path) or self
