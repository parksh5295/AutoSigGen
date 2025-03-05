import argparse


# Create an instance that can receive argument values
parser = argparse.ArgumentParser(description='Argparser')

# Set the argument values to be input (default value can be set)
parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
parser.add_argument('--file_number', type=int, default=1)   # Detach files
parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
parser.add_argument('--heterogeneous', type=str, default="Normalized")   # Heterogeneous(Embedding) Methods
parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
parser.add_argument('--association', type=str, default="apriori")   # Association Rule

# Save the above in args
args = parser.parse_args()

# Output the value of the input arguments
print(args.file_type)
print(args.file_number)
print(args.train_test)
print(args.heterogeneous)
print(args.clustering)
print(args.association)