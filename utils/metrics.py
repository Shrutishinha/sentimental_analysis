"""
=========================================================
Evaluation Metrics Utility
=========================================================

Functions Included
------------------
✔ Accuracy
✔ Precision
✔ Recall
✔ F1 Score
✔ ROC AUC
✔ Classification Report
✔ Confusion Matrix
✔ Cross Validation
✔ Model Evaluation
✔ Comparison Table

=========================================================
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.model_selection import cross_val_score


# -------------------------------------------------------
# Accuracy
# -------------------------------------------------------

def accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)


# -------------------------------------------------------
# Precision
# -------------------------------------------------------

def precision(y_true, y_pred):

    return precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# -------------------------------------------------------
# Recall
# -------------------------------------------------------

def recall(y_true, y_pred):

    return recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# -------------------------------------------------------
# F1 Score
# -------------------------------------------------------

def f1(y_true, y_pred):

    return f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


# -------------------------------------------------------
# ROC AUC
# -------------------------------------------------------

def roc_auc(y_true, probabilities):

    try:

        return roc_auc_score(
            y_true,
            probabilities
        )

    except:

        return None


# -------------------------------------------------------
# Classification Report
# -------------------------------------------------------

def report(y_true, y_pred):

    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )


# -------------------------------------------------------
# Report DataFrame
# -------------------------------------------------------

def report_dataframe(y_true, y_pred):

    rep = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(rep).transpose()


# -------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------

def get_confusion_matrix(y_true, y_pred):

    return confusion_matrix(
        y_true,
        y_pred
    )


# -------------------------------------------------------
# MAE
# -------------------------------------------------------

def mae(y_true, y_pred):

    return mean_absolute_error(
        y_true,
        y_pred
    )


# -------------------------------------------------------
# RMSE
# -------------------------------------------------------

def rmse(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return np.sqrt(mse)


# -------------------------------------------------------
# Cross Validation
# -------------------------------------------------------

def cross_validation(
    model,
    X,
    y,
    cv=5
):

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    return {

        "scores": scores,

        "mean": scores.mean(),

        "std": scores.std()

    }


# -------------------------------------------------------
# Complete Evaluation
# -------------------------------------------------------

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    results = {

        "Accuracy":
        accuracy(y_test, predictions),

        "Precision":
        precision(y_test, predictions),

        "Recall":
        recall(y_test, predictions),

        "F1 Score":
        f1(y_test, predictions),

        "Confusion Matrix":
        get_confusion_matrix(
            y_test,
            predictions
        ),

        "Classification Report":
        report_dataframe(
            y_test,
            predictions
        )

    }

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(X_test)

        if probability.shape[1] == 2:

            results["ROC AUC"] = roc_auc(
                y_test,
                probability[:, 1]
            )

    return results


# -------------------------------------------------------
# Compare Multiple Models
# -------------------------------------------------------

def compare_models(results):

    """
    Example Input

    [
        {
            "Model":"Logistic Regression",
            "Accuracy":0.91,
            "Precision":0.90,
            "Recall":0.91,
            "F1":0.90
        }
    ]
    """

    df = pd.DataFrame(results)

    return df.sort_values(
        by="Accuracy",
        ascending=False
    )


# -------------------------------------------------------
# Print Metrics
# -------------------------------------------------------

def print_metrics(metrics):

    print("=" * 60)

    for key, value in metrics.items():

        if isinstance(value, float):

            print(f"{key:<20}: {value:.4f}")

        else:

            print(f"{key:<20}:")
            print(value)

    print("=" * 60)


# -------------------------------------------------------
# Sample Test
# -------------------------------------------------------

if __name__ == "__main__":

    y_true = [1, 0, 1, 1, 0, 1]

    y_pred = [1, 0, 1, 0, 0, 1]

    print("Accuracy :", accuracy(y_true, y_pred))

    print("Precision :", precision(y_true, y_pred))

    print("Recall :", recall(y_true, y_pred))

    print("F1 :", f1(y_true, y_pred))

    print()

    print(report_dataframe(y_true, y_pred))

    print()

    print(get_confusion_matrix(y_true, y_pred))