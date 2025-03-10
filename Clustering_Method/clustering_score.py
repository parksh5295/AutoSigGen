from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
import numpy as np


def accuracy_basic(t, p):
    return{
        "accuracy": accuracy_score(t, p)
    }

def precision_basic(t, p, average):
    return{
        "precision": precision_score(t, p, average=average, zero_division=0)
    }

def recall_basic(t, p, average):
    return{
        "recall": recall_score(t, p, average=average, zero_division=0)
    }

def f1_basic(t, p, average):
    return{
        "f1_score": f1_score(t, p, average=average, zero_division=0)
    }

def jaccard_basic(t, p, average):
    return{
        "jaccard": jaccard_score(t, p, average=average, zero_division=0)
    }

def silhouette_basic(x_data, p):
    return{
        "silhouette": silhouette_score(x_data, p) if len(set(p)) > 1 else np.nan
    }

def average_combination(t, p, average, x_data):
    return{
        accuracy_basic(t, p),
        precision_basic(t, p, average),
        recall_basic(t, p, average),
        f1_basic(t, p, average),
        jaccard_basic(t, p, average),
        silhouette_basic(x_data, p)
    }

def average_combination_wos(t, p, average):
    return{
        accuracy_basic(t, p),
        precision_basic(t, p, average),
        recall_basic(t, p, average),
        f1_basic(t, p, average),
        jaccard_basic(t, p, average)
    }

def evaluate_clustering(y_true, y_pred, X_data):
    if not y_true.empty:
        return {
            "average=macro": average_combination(y_true, y_pred, 'macro', X_data),
            "average=micro": average_combination(y_true, y_pred, 'micro', X_data),
            "average=weighted": average_combination(y_true, y_pred, 'micro', X_data)
        }
    return {}

def evaluate_clustering_wos(y_true, y_pred):
    if not y_true.empty:
        return {
            "average=macro": average_combination_wos(y_true, y_pred, 'macro'),
            "average=micro": average_combination_wos(y_true, y_pred, 'micro'),
            "average=weighted": average_combination_wos(y_true, y_pred, 'micro')
        }
    return {}