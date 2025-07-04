import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import config
from signals.model_preparation import prepare_volatility_data


class VolatilityModel:
    def __init__(self):
        self.stage1_pipeline = None
        self.final_model = None
        self.selected_features = []
        self.importances = {}
        self.last_trained = None  # datetime object

    def train_stage1(self, df, importance_threshold=0.01, n_splits=3):
        try:
            X, y, feature_cols = prepare_volatility_data(df, future_window=45, threshold=0.003)
        except Exception as e:
            print("[VolTrain][Error] Failed during feature preparation:")
            import traceback; traceback.print_exc()
            raise e

        cutoff_date = X.index.max() - pd.Timedelta(days=30)
        X_old = X[X.index < cutoff_date]
        y_old = y[X.index < cutoff_date]

        if len(X_old) < 4:
            print("[VolTrain][Warning] Not enough historical samples. Using full dataset.")
            X_old, y_old = X.copy(), y.copy()

        best_score = 0
        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X_old) - 1))

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_old)):
            try:
                X_train, X_test = X_old.iloc[train_idx], X_old.iloc[test_idx]
                y_train, y_test = y_old.iloc[train_idx], y_old.iloc[test_idx]

                preprocessor = ColumnTransformer([
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), feature_cols)
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
                auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
                print(f"[VolTrain][Fold {fold + 1}] AUC: {auc:.4f}")

                if auc > best_score:
                    best_score = auc
                    self.stage1_pipeline = pipeline
                    importances = model.feature_importances_
                    idx = np.where(importances >= importance_threshold)[0]
                    self.selected_features = [feature_cols[i] for i in idx]
                    self.importances = {self.selected_features[i]: float(importances[idx[i]]) for i in range(len(idx))}

            except Exception as e:
                print(f"[VolTrain][Error] Fold {fold+1} failed:")
                import traceback; traceback.print_exc()

        if not self.selected_features:
            raise RuntimeError("[VolTrain][Critical] No features selected after training.")

        self.last_trained = datetime.utcnow()  # record training time
        print(f"[VolTrain] Selected {len(self.selected_features)} features.")
        return self

    def retrain_stage2(self, df):
        try:
            X, y, _ = prepare_volatility_data(df, future_window=45, threshold=0.003)

            cutoff_date = X.index.max() - pd.Timedelta(days=30)
            X_recent = X[X.index >= cutoff_date]
            y_recent = y[X.index >= cutoff_date]

            valid_features = [f for f in self.selected_features if f in X_recent.columns]
            if not valid_features:
                raise ValueError("[VolTrain] None of the selected features are in recent data.")

            classifier = clone(self.stage1_pipeline.named_steps['classifier'])
            preprocessor = ColumnTransformer([
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), valid_features)
            ])

            self.final_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])

            self.final_model.fit(X_recent[valid_features], y_recent)
            print("[VolTrain] Final model retrained successfully.")

            self.last_trained = datetime.utcnow()  # update training time after stage 2 retrain

        except Exception as e:
            print("[VolTrain][Error] Failed in retrain_stage2:")
            import traceback; traceback.print_exc()
            raise e

        return self

    def predict(self, df_latest):
        if not self.final_model:
            raise RuntimeError("[VolPredict] Final model not available.")

        try:
            missing = set(self.selected_features) - set(df_latest.columns)
            if missing:
                print(f"[VolPredict][Warning] Filling missing columns: {missing}")
                for col in missing:
                    df_latest[col] = 0.0

            X_latest = df_latest[self.selected_features].copy()
            if X_latest.isnull().any().any():
                raise ValueError(f"[VolPredict] NaNs found in input columns: {X_latest.columns[X_latest.isnull().any()].tolist()}")

            return self.final_model.predict(X_latest)[0], self.final_model.predict_proba(X_latest)[0][1]

        except Exception as e:
            print("[VolPredict][Error] Prediction failed:")
            import traceback; traceback.print_exc()
            raise e

    def save(self, model_path=config.VOL_MODEL_FILE, feature_path=config.VOL_FEATURES_FILE):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)

            joblib.dump(self.final_model, model_path)
            metadata = {
                "features": self.selected_features,
                "importances": self.importances,
                "sklearn_version": pd.__version__,
                "xgboost_version": xgb.__version__,
                "last_trained": self.last_trained.isoformat() if self.last_trained else None
            }
            with open(feature_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"[VolSave] Model saved to {model_path}")
            print(f"[VolSave] Feature metadata saved to {feature_path}")
        except Exception as e:
            print("[VolSave][Error] Failed to save model or metadata:")
            import traceback; traceback.print_exc()
            raise e

    def load(self, model_path=config.VOL_MODEL_FILE, feature_path=config.VOL_FEATURES_FILE):
        if not os.path.exists(model_path) or not os.path.exists(feature_path):
            raise FileNotFoundError("[VolLoad] Missing model or metadata file.")

        try:
            self.final_model = joblib.load(model_path)
            with open(feature_path, "r") as f:
                metadata = json.load(f)
                self.selected_features = metadata["features"]
                self.importances = metadata.get("importances", {})
                last_trained_str = metadata.get("last_trained", None)
                self.last_trained = datetime.fromisoformat(last_trained_str) if last_trained_str else None

            print(f"[VolLoad] Model loaded with {len(self.selected_features)} features.")
            if self.last_trained:
                print(f"[VolLoad] Last trained: {self.last_trained.isoformat()}")
        except Exception as e:
            print("[VolLoad][Error] Failed to load model or metadata:")
            import traceback; traceback.print_exc()
            raise e

        return self