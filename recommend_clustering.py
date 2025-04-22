# clustering_recommender.py

import argparse
import time
import os
import pandas as pd
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import silhouette_score
from scipy.stats import kurtosis


def recommend_clustering_by_distribution(X_df):
    X = X_df.values
    n_samples, n_features = X.shape
    stats = {'n_samples': n_samples, 'n_features': n_features}

    # GMM covariance ratio
    reg_value = 1e-6 # Add regularization for numerical stability
    try:
        gmm_diag = GaussianMixture(n_components=2, covariance_type='diag', random_state=0, reg_covar=reg_value).fit(X)
        gmm_full = GaussianMixture(n_components=2, covariance_type='full', random_state=0, reg_covar=reg_value).fit(X)
        diag_ratios = [np.sum(np.diag(f)) / np.sum(f) for f in gmm_full.covariances_]
        avg_diag_ratio = np.mean(diag_ratios)
    except np.linalg.LinAlgError as e:
        print(f"[Warning] GMM fitting failed due to LinAlgError: {e}. Using default avg_diag_ratio = 0.5")
        avg_diag_ratio = 0.5 # Assign a default value or handle differently
    except ValueError as e:
        print(f"[Warning] GMM fitting failed due to ValueError: {e}. Using default avg_diag_ratio = 0.5")
        avg_diag_ratio = 0.5 # Handle cases like insufficient samples

    stats['covariance_diagonal_ratio'] = avg_diag_ratio

    est_n_clusters = estimate_clusters(X)
    metrics = {'estimated_clusters': est_n_clusters}

    # Recommendation algorithm limit
    recommendations = []

    if avg_diag_ratio > 0.85:
        recommendations += ['GMM', 'KMeans', 'SGMM']
    elif avg_diag_ratio > 0.65:
        recommendations += ['GK', 'GMM', 'KMeans']
    else:
        recommendations += ['DBSCAN', 'MeanShift', 'FCM']

    recommendations += ['NeuralGas']

    reason_summary = f"""
        [Distribution-Based Recommendation]
        - Covariance Diagonal Ratio: {avg_diag_ratio:.2f}
        - Estimated #Clusters: {est_n_clusters}
        - Sample/Cluster Ratio: {n_samples / est_n_clusters:.2f}
        """

    return recommendations, metrics, stats, reason_summary.strip()


def recommend_clustering_by_feature_types(X_df):
    X = X_df.values
    n_samples, n_features = X.shape
    stats = {'n_samples': n_samples, 'n_features': n_features}
    metrics = {'estimated_clusters': min(5, max(2, n_samples // 100))}

    # Recommend simple and stable clustering methods (within specified 11)
    recommendations = ['KMeans', 'K-Medians', 'GMeans', 'NeuralGas']

    reason_summary = f"""
        [Feature-Type-Based Recommendation]
        - Sample Size: {n_samples}, Feature Dim: {n_features}
        - Estimated #Clusters: {metrics['estimated_clusters']}
        - Using simple clustering methods for small or mixed feature data
        """

    return recommendations, metrics, stats, reason_summary.strip()


def estimate_clusters(X):
    # Simple cluster number estimation (silhouette optimization)
    best_score = -1
    best_k = 2
    for k in range(2, min(10, len(X))):
        try:
            labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    return best_k



def smart_clustering_selector(X_df, raw_df, threshold_sample=500, threshold_dim=10):
    n_samples, n_features = X_df.shape
    print(f"\n[Info] Sample size: {n_samples}, Feature dim: {n_features}")

    if n_samples < threshold_sample or n_features < threshold_dim:
        print("[Strategy] Using feature-type-based recommendation (small data).")
        algorithms, metrics, stats, reason_summary = recommend_clustering_by_feature_types(X_df)
    else:
        print("[Strategy] Using distribution-based recommendation.")
        try:
            algorithms, metrics, stats, reason_summary = recommend_clustering_by_distribution(X_df)
        except Exception as e:
            print(f"[Fallback] Distribution-based failed due to: {e}")
            algorithms, metrics, stats, reason_summary = recommend_clustering_by_feature_types(X_df)

    return algorithms, metrics, stats, reason_summary


def explain_recommendation(algorithm, metrics, stats):
    explanation = f"[Recommendation] {algorithm}\n"

    if 'covariance_diagonal_ratio' in stats:
        diag_ratio = stats['covariance_diagonal_ratio']
        explanation += f"[Recommendation Reason] The average covariance matrix has a diagonal ratio of {diag_ratio:.2f}, which suggests that GMM(diagonal) can effectively describe the data distribution.\n"

    if 'estimated_clusters' in metrics:
        n_clusters = metrics['estimated_clusters']
        explanation += f"[Recommendation Reason] The estimated number of clusters is {n_clusters}.\n"

    if 'n_samples' in stats and 'estimated_clusters' in metrics:
        ratio = metrics['estimated_clusters'] / stats['n_samples']
        explanation += f"[Recommendation Reason] The ratio of the number of samples to the number of clusters is {ratio:.4f}.\n"

    return explanation


def save_recommendation_to_csv(file_type, file_number, algorithms, metrics, stats, reason_summary):
    output_dir = f"../Dataset/recommend/"
    os.makedirs(output_dir, exist_ok=True)

    file_type_dir = os.path.join(output_dir, file_type)
    os.makedirs(file_type_dir, exist_ok=True)

    save_path = os.path.join(file_type_dir, f"{file_type}_{file_number}_recommendation.csv")

    df = pd.DataFrame({
        "Recommended_Algorithms": algorithms,
        "Best_Algorithm": [algorithms[0]] * len(algorithms),
        "Estimated_Clusters": [metrics.get("estimated_clusters", None)] * len(algorithms),
        "Cov_Diagonal_Ratio": [stats.get("covariance_diagonal_ratio", None)] * len(algorithms),
        "n_Samples": [stats.get("n_samples", None)] * len(algorithms),
        "n_Features": [stats.get("n_features", None)] * len(algorithms),
        "Reason_Summary": [reason_summary] * len(algorithms),
    })

    df.to_csv(save_path, index=False)
    print(f"\nSaved recommendation result to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Recommend clustering method for given dataset")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")
    parser.add_argument('--file_number', type=int, default=1)
    args = parser.parse_args()

    file_type = args.file_type
    file_number = args.file_number

    total_start = time.time()

    # 1. Load data
    file_path, _ = file_path_line_nonnumber(file_type, file_number)
    cut_type = 'random' if file_type in ['DARPA98', 'DARPA', 'NSL-KDD', 'NSL_KDD'] else 'all'
    data = file_cut(file_type, file_path, cut_type)

    # Clean column names by stripping leading/trailing whitespace
    data.columns = data.columns.str.strip()

    # 2. Apply label
    if file_type in ['MiraiBotnet', 'NSL-KDD']:
        data['label'], _ = anomal_judgment_nonlabel(file_type, data)
    elif file_type == 'netML':
        data['label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    elif file_type == 'DARPA98':
        data['label'] = data['Class'].apply(lambda x: 0 if x == '-' else 1)
    else:
        data['label'] = anomal_judgment_label(data)

    # 3. Time + embedding
    data = time_scalar_transfer(data, file_type)
    embedded_df, feature_list, category_mapping, data_list = choose_heterogeneous_method(
        data, file_type, het_method="Interval_inverse", regul='N'
    )
    X, _ = map_intervals_to_groups(embedded_df, category_mapping, data_list, regul='N')

    # 4. Recommend clustering method
    recommendations, metrics, stats, recommendation_explanation = smart_clustering_selector(X, data)

    print("\nRecommended Clustering Algorithms:")
    for i, algo in enumerate(recommendations, 1):
        print(f"  {i}. {algo}")

    print(f"\nExplanation:\n{recommendation_explanation}")

    # 5. Save to CSV
    save_recommendation_to_csv(file_type, file_number, recommendations, metrics, stats, recommendation_explanation)

    print(f"\nDone in {time.time() - total_start:.2f} seconds.")


if __name__ == '__main__':
    main()
