# Methods for choosing the best clustering method for your dataset

import argparse
import time
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judgment_nonlabel, anomal_judgment_label
from utils.time_transfer import time_scalar_transfer
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups
from Clustering_Method.recommend_clustering_method import recommend_clustering_by_distribution, recommend_clustering_by_feature_types


def smart_clustering_selector(X_df, raw_df, threshold_sample=500, threshold_dim=10):
    n_samples, n_features = X_df.shape
    print(f"\n[Info] Sample size: {n_samples}, Feature dim: {n_features}")

    if n_samples < threshold_sample or n_features < threshold_dim:
        print("[Strategy] Using feature-type-based recommendation (small data).")
        return recommend_clustering_by_feature_types(X_df)
    else:
        print("[Strategy] Using distribution-based recommendation.")
        try:
            return recommend_clustering_by_distribution(X_df)
        except Exception as e:
            print(f"[Fallback] Distribution-based failed due to: {e}")
            return recommend_clustering_by_feature_types(X_df)


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
    cut_type = 'random' if file_type in ['DARPA98', 'DARPA', 'NSL-KDD'] else 'all'
    data = file_cut(file_type, file_path, cut_type)

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
        data, file_type, heterogeneous_method="Interval_inverse", regul='N'
    )
    X, _ = map_intervals_to_groups(embedded_df, category_mapping, data_list, regul='N')

    # 4. Recommend clustering method
    recommendations = smart_clustering_selector(X, data)

    print("\n Recommended Clustering Algorithms:")
    for i, algo in enumerate(recommendations, 1):
        print(f"  {i}. {algo}")

    print(f"\n Done in {time.time() - total_start:.2f} seconds.")


if __name__ == '__main__':
    main()
