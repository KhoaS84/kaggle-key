"""
Evaluation and Metrics module.
"""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Calculate classification accuracy and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "confusion_matrix": cm}
