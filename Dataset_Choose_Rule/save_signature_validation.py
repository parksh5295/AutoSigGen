import pandas as pd

def save_validation_results(file_type, file_number, association_rule, basic_eval, fp_results, overfit_results):
    """
    Save all evaluation results in a single CSV file
    """
    save_path = f"../Dataset/validation/{file_type}/"
    ensure_directory_exists(save_path)

    # Combine all results into a single dictionary
    validation_results = {
        'Basic_Evaluation': basic_eval,
        'False_Positive_Analysis': fp_results,
        'Overfitting_Analysis': overfit_results
    }

    # Convert results to DataFrame and save
    validation_df = pd.DataFrame([validation_results])
    validation_path = f"{save_path}{file_type}_{association_rule}_{file_number}_validation_results.csv"
    validation_df.to_csv(validation_path, index=False)
    print(f"All validation results saved to: {validation_path}")
