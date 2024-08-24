import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
import numpy as np

# Class for code clustering
class CodeClusterer:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.data = None
        self.model = KMeans(n_clusters=num_clusters, random_state=42)
        self.elbow_scores = []
        self.silhouette_avg = None
        self.davies_bouldin = None

    def load_data(self, dataframe):
        self.data = dataframe

    def cluster_codes(self):
        if self.data is None or self.data.empty:
            raise ValueError("DataFrame is empty or not loaded.")
        features = self.data[['Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity']]
        self.model.fit(features)
        self.data['Cluster'] = self.model.labels_
        self.silhouette_avg = silhouette_score(features, self.model.labels_)
        self.davies_bouldin = davies_bouldin_score(features, self.model.labels_)
        return features

    def get_clustered_data(self):
        return self.data

    def calculate_elbow(self, max_clusters=10):
        if self.data is None or self.data.empty:
            raise ValueError("DataFrame is empty or not loaded.")
        features = self.data[['Text_Similarity', 'Structural_Similarity', 'Weighted_Similarity']]
        for i in range(2, max_clusters + 1):
            model = KMeans(n_clusters=i, random_state=42)
            model.fit(features)
            self.elbow_scores.append(model.inertia_)

    def get_silhouette_data(self, features):
        return pd.DataFrame({
            'Cluster': self.data['Cluster'],
            'Silhouette Value': silhouette_samples(features, self.data['Cluster'])
        })

# Function to find the elbow point in the elbow scores
def find_elbow_point(elbow_scores):
    if not elbow_scores:
        return 2
    changes = np.diff(elbow_scores)
    return np.argmin(changes) + 2
