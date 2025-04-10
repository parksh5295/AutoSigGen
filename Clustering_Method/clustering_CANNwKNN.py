# Clustering Algorithm: CANN(Clustering Assisted by Nearest Neighbors)
# input 'X' is X_reduced or X rows
# (pre)Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)
# (main)Return: dictionary{Cluster Information(0, 1 Classification), best_parameter_dict}

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense, Flatten, Attention
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from utils.progressing_bar import progress_bar
from utils.class_row import nomal_class_data
from Tuning_hyperparameter.Grid_search import Grid_search_all


# Defining the CANN model
class CANN(tf.keras.Model):
    def __init__(self, input_shape):
        super(CANN, self).__init__()
        self.dense1 = Dense(64, activation='relu', input_shape=input_shape)
        self.flatten = Flatten()
        self.dense2 = Dense(32, activation='relu')
        self.dense3 = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.flatten(x)
        x = self.dense2(x)
        return self.dense3(x)
    

def clustering_CANNwKNN(data, X):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Define model input shapes
        input_shape = (X.shape[1],)

        benign_data = nomal_class_data(data) # Assuming that we only know benign data

        # Prepare data
        X_benign = benign_data.drop(columns=['label'], errors='ignore').to_numpy()
        y_benign = np.zeros(len(X_benign))  # benign is considered label 0

        # Create & tune model
        tune_parameters = Grid_search_all(X, 'CANNwKNN', None, data)  # ONLY benign!
        best_params = tune_parameters['CANNwKNN']['best_params']
        parameter_dict = tune_parameters['CANNwKNN']['all_params']
        parameter_dict.update(best_params)

        # Train CANN on benign only
        cann = create_cann_model(input_shape)
        cann.fit(X_benign, y_benign, epochs=parameter_dict['epochs'], batch_size=parameter_dict['batch_size'])

        # Feature extraction
        features_all = cann.predict(X)  # Extract for all data
        features_benign = cann.predict(X_benign)

        # Train KNN on benign features only
        knn = KNeighborsClassifier(n_neighbors=parameter_dict['n_neighbors'])
        knn.fit(features_benign, y_benign)

        # Predict for all data
        predicted = knn.predict(features_all)

        # Save cluster labels
        data['cluster'] = predicted  # Determined as benign=0, unknown→0/1

        update_pbar(len(data))

    predict_CANNwKNN = data['cluster']

    return {
        'Cluster_labeling': predict_CANNwKNN,
        'Best_parameter_dict': parameter_dict
    }
    

# Function to wrap CANN in a sklearn-compatible classifier
def create_cann_model(input_shape):
    model = CANN(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define a wrapper class for GridSearchCV
class CANNWithKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, epochs=100, batch_size=32, n_neighbors=5):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.model = None
        self.knn = None

    def fit(self, X, y):
        # Create and train the CANN model
        self.model = create_cann_model(self.input_shape)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

        # Extract features from the trained model
        features = self.model.predict(X)

        # Apply KNN on the extracted features
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(features, y)

        return self

    def predict(self, X):
        # Use trained KNN to predict clusters
        features = self.model.predict(X)
        return self.knn.predict(features)
    
    def fit_predict(self, X, data):
        # Step 1: Get benign data from full dataframe
        benign_data = nomal_class_data(data)
        X_benign = benign_data.drop(columns=['label']).to_numpy()
        y_benign = np.zeros(len(X_benign))  # All benign = 0

        # Step 2: Fit using benign only
        self.fit(X_benign, y_benign)

        # Step 3: Predict for all data
        return self.predict(X)
    

def pre_clustering_CANNwKNN(data, X, epochs, batch_size, n_neighbors):
    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Define model input shapes
        input_shape = (X.shape[1],)

        # Create CANN Model
        model = CANN(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Learning CANN Model
        model.fit(X, data['label'], epochs, batch_size)

        # Feature Extraction with CANN Models
        features = model.predict(X)

        # Apply K-NN clustering
        knn = KNeighborsClassifier(n_neighbors)
        # The number of nearest neighbors (K points) to reference when classifying new data points
        knn.fit(features, data['label'])

        # Predict
        predictions = knn.predict(features)

        update_pbar(len(data))

    predict_CANNwKNN = predictions

    return {
        'Cluster_labeling' : predict_CANNwKNN,
        'n_clusters' : 2,
        'before_labeling' : predictions
    }