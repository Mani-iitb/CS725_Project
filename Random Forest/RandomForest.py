from DecisionTree import DecisionTree
from collections import Counter
import numpy as np


class RandomForest:
    def __init__(self, n_trees = 2, max_depth=100, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth = self.max_depth, 
                          min_samples_split=self.min_samples_split,
                          n_features =  self.n_features)
            X_sample, y_sample = self.sample_create(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def sample_create(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _most_common_label(self, y):
        counter = Counter(y)
        value =counter.most_common(1)[0][0]
        return value
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions,0,1)
        predictions = np.array([self._most_common_label(pred) for pred in predictions])
        return predictions
    