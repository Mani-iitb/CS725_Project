from DecisionTree import DecisionTreeClassifier
from collections import Counter
import numpy as np


class RandomForest:
    def __init__(self, num_trees=2, max_tree_depth=100, min_split_samples=2):
        self.num_trees = num_trees
        self.max_tree_depth = max_tree_depth
        self.min_split_samples = min_split_samples
        self.tree_list = []
        
    def fit(self, features, targets):
        self.tree_list = []
        self.feature_importances = []
        for _ in range(self.num_trees):
            tree_instance = DecisionTreeClassifier(max_depth=self.max_tree_depth, min_split=self.min_split_samples)
            sampled_features, sampled_targets = self.generate_samples(features, targets)
            tree_instance.fit(sampled_features, sampled_targets)
            self.tree_list.append(tree_instance)
            self.feature_importances.append(tree_instance.feature_importances_)
            
    def generate_samples(self, features, targets):
        num_samples = features.shape[0]
        random_indices = np.random.choice(num_samples, num_samples, replace=True)
        return features[random_indices], targets[random_indices]

    def _most_frequent_label(self, labels):
        label_counts = Counter(labels)
        most_common_label = label_counts.most_common(1)[0][0]
        return most_common_label
    
    def predict(self, features):
        tree_predictions = np.array([tree.predict(features) for tree in self.tree_list])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        final_predictions = np.array([self._most_frequent_label(tree_pred) for tree_pred in tree_predictions])
        return final_predictions
