# A program that automatically generates signatures using association rules.

import argparse
import numpy as np
from Dataset_Choose_Rule.choose_amount_dataset import file_path_line_nonnumber, file_cut
from definition.Anomal_Judgment import anomal_judment_label, anomal_judgment_nonlabel
from utils.class_row import anomal_class_data, without_labelmaking_out, nomal_class_data
from Modules.Heterogeneous_module import choose_heterogeneous_method
from Modules.Association_module import association_module
from Modules.Signature_evaluation_module import signature_evaluate
from Modules.Signature_underlimit import under_limit
from Evaluation.calculate_signature import calculate_signatures
from Modules.Difference_sets import dict_list_difference


def main():
    # argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Normalized")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--eval_clustering_silhouette', type=str, default="n")
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule
    parser.add_argument('--precision_underlimit', type=float, default=0.7)
    parser.add_argument('--signature_ea', type=int, default=5)

    # Save the above in args
    args = parser.parse_args()

    # Output the value of the input arguments
    file_type = args.file_type
    file_number = args.file_number
    train_tset = args.train_test
    heterogeneous_method = args.heterogeneous
    clustering_algorithm = args.clustering
    eval_clustering_silhouette = args.eval_clustering_silhouette
    Association_mathod = args.association
    precision_underlimit = args.precision_underlimit
    signature_ea = args.signature_ea


    # 1. Data loading
    file_path = file_path_line_nonnumber(file_type, file_number)
    cut_type = str(input("Enter the data cut type: "))
    data = file_cut(file_path, cut_type)


    # 2. Handling judgments of Anomal or Nomal
    if file_type == 'MiraiBotnet':
        data['label'] = anomal_judgment_nonlabel(file_type, data)
    else:
        data['label'] = anomal_judment_label(data)

    anomal_rows = anomal_class_data(data)
    anomal_data = without_labelmaking_out(file_type, anomal_rows)

    # Information about how to set up association rule groups
    anomal_grouped_data, fl = choose_heterogeneous_method(anomal_data, file_type, heterogeneous_method)
    # anomal_grouped_data is DataFrame
    # fl: feature list; Same contents but not used because it's not inside a DF.

    # Make nomal row
    nomal_rows = nomal_class_data(data)
    nomal_data = without_labelmaking_out(file_type, nomal_rows)

    nomal_grouped_data, flo = choose_heterogeneous_method(nomal_data, file_type, heterogeneous_method)
    # nomal_grouped_data is DataFrame
    # flo: feature list; Same contents but not used because it's not inside a DF.

    data_without_label = without_labelmaking_out(file_type, data)
    data_without_label, f_w = choose_heterogeneous_method(data_without_label, file_type, heterogeneous_method)
    data_without_label['label'] = data['label']
    heterogeneous_whole_data = data_without_label


    # 3. Set association statements (confidence ratios, etc.)
    # I need to let them choose if they want confidence to be selected automatically.
    min_support = 0.1
    best_confidence = 0.8    # Initialize the variables to change
    # Considering anomalies and nomals simultaneously

    confidence_values = np.arange(0.1, 1.0, 0.05)
    best_recall = 0


    # Identify the signatures with the highest recall in your situation
    # 4. Excute Association Rule, Manage related groups
    for min_confidence in confidence_values:
        association_list_anomal = association_module(anomal_grouped_data, Association_mathod, min_support, min_confidence)

        
        # 5. Find a difference-set association group
        association_list_nomal = association_module(nomal_grouped_data, Association_mathod, min_support, min_confidence)

        # A collection of pure anomalous signatures created with difference_set
        signatures = dict_list_difference(association_list_anomal, association_list_nomal)


        # 6. Make Signature
        signature_result = signature_evaluate(data, signatures) # Evaluation and score of each signature is available, LIST (internal DICT)
        signature_sets = under_limit(signature_result, signature_ea, precision_underlimit)  # Collection of signatures before validating recall


        # 7. Evaluate association rule (Signature)
        current_recall = calculate_signatures(data, signature_sets)  # Score of the final signature collection
        # 

        # Update the highest Recall value
        if current_recall > best_recall:
            best_recall = current_recall
            best_confidence = min_confidence
            last_signature_sets = signature_sets    # signatures in best_confidence


    association_result = {
        'Verified_Signatures': last_signature_sets,
        'Recall': best_recall,
        'Best_confidence': best_confidence
    }
    print(association_result)


    return association_result


if __name__ == '__main__':
    main()