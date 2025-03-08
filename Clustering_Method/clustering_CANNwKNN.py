# Clustering Algorithm: CANN(Clustering Assisted by Nearest Neighbors)
# input 'X' is X_reduced or X rows
# Return: Cluster Information(0, 1 Classification), num_clusters(result), Cluster Information(not fit, Non-classification, optional)

# This file NEED to repair !!!

from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense, Flatten, Attention
import tensorflow as tf
from utils.progressing_bar import progress_bar


# Defining the CANN model
class CANN(tf.keras.Model):
    def __init__(self, input_shape):
        super(CANN, self).__init__()
        self.dense1 = Dense(64, activation='relu', input_shape=input_shape)
        self.attention = Attention()
        self.flatten = Flatten()
        self.dense2 = Dense(32, activation='relu')
        self.dense3 = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.attention([x, x])  # Applying the Attention mechanism
        x = self.flatten(x)
        x = self.dense2(x)
        return self.dense3(x)
    

def clustering_CANNwKNN(data, X, epochs, batch_size):
    # Define model input shapes
    input_shape = (X.shape[1],)

    # Create CANN Model
    model = CANN(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Learning CANN Model
    model.fit(X, data['label'], epochs=epochs, batch_size=batch_size)

    with progress_bar(len(data), desc="Clustering", unit="samples") as update_pbar:
        # Feature Extraction with CANN Models
        features = model.predict(X)

        # Apply K-NN clustering
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(features, data['label'])

        # Predict
        predictions = knn.predict(features)
        data['cluster'] = predictions

        update_pbar(len(data))
    