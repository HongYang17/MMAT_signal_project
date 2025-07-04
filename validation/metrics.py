import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, auc
)


def evaluate_model(pipeline, X, y, title="Model Evaluation", save_path=None):
    print("\n=== Evaluation Report ===")
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    print_classification_metrics(y, y_pred, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_path:
        plt.savefig(save_path + "_confusion_matrix.png", dpi=300)
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"ROC Curve - {title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path + "_roc_curve.png", dpi=300)
    plt.show()

def print_classification_metrics(y_true, y_pred, confidences=None):
    print("\n[Eval] Classification Report:")

    # Convert y_true boolean hits to directional string labels if needed
    if pd.api.types.is_bool_dtype(y_true) and pd.api.types.is_string_dtype(y_pred):
        print("[Info] Detected boolean y_true and string y_pred. Converting y_true into directional labels...")
        y_true = pd.Series([
            pred if hit else ("DOWN" if pred == "UP" else "UP")
            if pred in ("UP", "DOWN") else "NEUTRAL"
            for hit, pred in zip(y_true, y_pred)
        ], index=y_true.index if hasattr(y_true, 'index') else None)

    # Remove NEUTRAL and invalid predictions
    mask = pd.Series(y_pred).isin(["UP", "DOWN"]) & pd.Series(y_true).isin(["UP", "DOWN"])
    y_true_filtered = pd.Series(y_true)[mask]
    y_pred_filtered = pd.Series(y_pred)[mask]
    conf_filtered = pd.Series(confidences)[mask] if confidences is not None else None

    # Print classification metrics
    print(classification_report(y_true_filtered, y_pred_filtered, zero_division=0))
    print(f"Accuracy:  {accuracy_score(y_true_filtered, y_pred_filtered):.4f}")
    print(f"Precision: {precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0):.4f}")

    if conf_filtered is not None:
        try:
            print(f"AUC:       {roc_auc_score((y_true_filtered == 'UP').astype(int), conf_filtered):.4f}")
        except Exception as e:
            print(f"[Warning] AUC computation failed: {e}")