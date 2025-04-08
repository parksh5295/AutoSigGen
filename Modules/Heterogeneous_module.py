# Modules for bundling and using Heterogeneous_Methods
# Return: (embedded) data, feature_list

from Heterogeneous_Method.Non_act import Heterogeneous_Non_OneHotEncoder, Heterogeneous_Non_StandardScaler
from Heterogeneous_Method.Interval_Same import Heterogeneous_Interval_Same
from Heterogeneous_Method.Interval_normalized import Heterogeneous_Interval_Inverse


def choose_heterogeneous_method(data, file_type, het_method='Interval_inverse', regul='N'):
    if het_method == 'Non_act':
        het_method_Non_act = str(input("Choose between OneHotEncoder and StandardScaler: "))
        if het_method_Non_act == 'OneHotEncoder':
            embedded_data, feature_list = Heterogeneous_Non_OneHotEncoder(data)
        elif het_method_Non_act == 'StandardScaler':
            embedded_data, feature_list = Heterogeneous_Non_StandardScaler(data)
        else:
            print("There are two choices: OneHotEncoder and StandardScaler. Please try again.")
            choose_heterogeneous_method(data, file_type, 'Non_act')
    elif het_method == 'Interval_same':
        embedded_data, feature_list = Heterogeneous_Interval_Same(data, file_type)
    elif het_method in ['Interval_inverse', 'Interval_Inverse']:
        embedded_data, feature_list, category_mapping = Heterogeneous_Interval_Inverse(data, file_type, regul)
    else:
        print("Invalid input: Please double-check your heterogenization method.")
        choose_heterogeneous_method(data, file_type, het_method)

    return embedded_data, feature_list, category_mapping
    # category_mapping: [0]->categorical features mapping, [1]->flag features mapping (Non Standard)