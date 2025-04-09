from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, silhouette_score
import numpy as np


def accuracy_basic(t, p):
    metric = accuracy_score(t, p)
    return metric

def precision_basic(t, p, average):
    metric = precision_score(t, p, average=average, zero_division=0)
    return metric

def recall_basic(t, p, average):
    metric = recall_score(t, p, average=average, zero_division=0)
    return metric

def f1_basic(t, p, average):
    metric = f1_score(t, p, average=average, zero_division=0)
    return metric

def jaccard_basic(t, p, average):
    metric = jaccard_score(t, p, average=average, zero_division=0)
    return metric

def silhouette_basic(x_data, p):
    metric = silhouette_score(x_data, p) if len(set(p)) > 1 else np.nan
    return metric


def average_combination(t, p, average, x_data):
    all_metrics = {
        "accuracy" : accuracy_basic(t, p),
        "precision" : precision_basic(t, p, average),
        "recall" : recall_basic(t, p, average),
        "f1" : f1_basic(t, p, average),
        "jaccard" : jaccard_basic(t, p, average),
        "silhouette" : silhouette_basic(x_data, p)
    }
    return all_metrics

def average_combination_wos(t, p, average):
    all_metrics = {
        "accuracy" : accuracy_basic(t, p),
        "precision" : precision_basic(t, p, average),
        "recall" : recall_basic(t, p, average),
        "f1" : f1_basic(t, p, average),
        "jaccard" : jaccard_basic(t, p, average)
    }
    return all_metrics


def evaluate_clustering(y_true, y_pred, X_data):
    if not y_true.empty:
        return {
            "average=macro": average_combination(y_true, y_pred, 'macro', X_data),
            "average=micro": average_combination(y_true, y_pred, 'micro', X_data),
            "average=weighted": average_combination(y_true, y_pred, 'weighted', X_data)
        }
    return {}

def evaluate_clustering_wos(y_true, y_pred):
    if not y_true.empty:
        return {
            "average=macro": average_combination_wos(y_true, y_pred, 'macro'),
            "average=micro": average_combination_wos(y_true, y_pred, 'micro'),
            "average=weighted": average_combination_wos(y_true, y_pred, 'weighted')
        }
    return {}