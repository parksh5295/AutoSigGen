import pandas as pd
import os
import json
import numpy as np


def ensure_directory_exists(filepath):
    """If the directory of the specified file path does not exist, create it."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)

# Helper function to convert numpy types
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist()) # Convert arrays to lists first
    elif pd.isna(obj): # Handle pandas NA/NaN specifically for JSON
        return None # Represent NaN as null in JSON
    return obj

def save_validation_results(file_type, file_number, association_rule, basic_eval, fp_results, overfit_results,
                            recall_before=None, recall_after=None, filtered_eval=None):
    """
    Save all evaluation results, including recall metrics and filtered eval, in a single CSV/JSON file
    """
    save_path = f"../Dataset/validation/{file_type}/"
    ensure_directory_exists(save_path)

    # Combine all results into a single dictionary
    validation_results = {
        'Basic_Evaluation': basic_eval.to_dict('records') if isinstance(basic_eval, pd.DataFrame) else basic_eval,
        'False_Positive_Analysis': fp_results.to_dict('records') if isinstance(fp_results, pd.DataFrame) else fp_results,
        'Overfitting_Analysis': overfit_results,
        'Recall_Before_FP_Removal': recall_before,
        'Recall_After_FP_Removal': recall_after,
        'Filtered_Signature_Evaluation': filtered_eval.to_dict('records') if isinstance(filtered_eval, pd.DataFrame) else filtered_eval
    }

    # --- Convert numpy types before saving --- 
    validation_results_serializable = convert_numpy_types(validation_results)
    # -----------------------------------------

    # Convert results to DataFrame and save (with JSON fallback)
    try:
        # Attempt to flatten the structure for CSV if possible, or save selectively
        # Saving complex nested structures directly to CSV via DataFrame often fails or is unreadable.
        # Consider saving key summary stats to CSV and full details to JSON.

        # Simple CSV save attempt (might fail or be messy)
        # Note: Converting the complex structure to a single-row DataFrame for CSV isn't ideal.
        # Consider creating a summary DataFrame instead if CSV is strictly needed.
        try:
            validation_df = pd.DataFrame([validation_results_serializable]) # Use serializable version
            validation_path = f"{save_path}{file_type}_{association_rule}_{file_number}_validation_results.csv"
            validation_df.to_csv(validation_path, index=False)
            print(f"Attempted to save validation results summary to CSV: {validation_path}")
        except Exception as csv_e:
            print(f"Warning: Could not save detailed results to CSV directly due to complex structure: {csv_e}")
            print("Saving full results reliably to JSON instead.")

        # Always save to JSON for reliable complex structure storage
        json_path = f"{save_path}{file_type}_{association_rule}_{file_number}_validation_results.json"
        with open(json_path, 'w') as f:
            json.dump(validation_results_serializable, f, indent=4) # Use serializable version
        print(f"Validation results reliably saved as JSON: {json_path}")

    except Exception as e:
        print(f"Error during saving validation results: {e}")
        # Fallback JSON save if initial try block failed before JSON save
        if 'json_path' in locals():
             try:
                 with open(json_path, 'w') as f:
                     # Use the already converted dictionary
                     json.dump(validation_results_serializable, f, indent=4) # Use serializable version
                 print(f"Fallback save successful as JSON: {json_path}")
             except Exception as json_e:
                 print(f"Error during fallback JSON save: {json_e}")
