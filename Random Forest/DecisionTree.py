import numpy as np
from collections import Counter

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, leaf_value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_value = leaf_value
        self.feature_importances_ = None
        
    def is_leaf_node(self):
        return self.leaf_value is not None

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_tree_depth=100):
        self.min_samples_split = min_samples_split
        self.max_tree_depth = max_tree_depth
        self.root_node = None

    def fit(self, feature_matrix, target_vector):
        num_features = len(feature_matrix[0])
        self.feature_importance_dict = {i: 0 for i in range(num_features)}
        self.root_node = self._grow_tree(feature_matrix, target_vector)
        self._calculate_feature_importances()

    def _grow_tree(self, feature_matrix, target_vector, depth=0):
        num_samples, num_features = feature_matrix.shape
        if num_samples == 0 or target_vector.shape[0] == 0:
            return DecisionTreeNode(leaf_value=-1)
        num_unique_labels = len(np.unique(target_vector))
        
        # Check for stopping conditions
        if depth >= self.max_tree_depth or num_unique_labels == 1 or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(target_vector)
            return DecisionTreeNode(leaf_value=leaf_value)
        
        # Randomly select feature indices
        selected_features = np.random.choice(num_features, num_features, replace=False)
        
        # Find the best split
        best_feature, best_threshold = self._determine_best_split(feature_matrix, target_vector, selected_features)
        
        # Partition the dataset
        left_indices, right_indices = self._split(feature_matrix[:, best_feature], best_threshold)
        
        # Recursively grow the left and right branches
        left_branch = self._grow_tree(feature_matrix[left_indices, :], target_vector[left_indices], depth + 1)
        right_branch = self._grow_tree(feature_matrix[right_indices, :], target_vector[right_indices], depth + 1)
        return DecisionTreeNode(feature=best_feature, threshold=best_threshold, left=left_branch, right=right_branch)
    
    def _determine_best_split(self, feature_matrix, target_vector, selected_features):
        best_gain = -1
        best_feature_index, best_threshold = None, None
        
        for feature_index in selected_features:
            feature_column = feature_matrix[:, feature_index]
            potential_thresholds = np.unique(feature_column)
            
            for threshold in potential_thresholds:
                information_gain = self._calculate_information_gain(target_vector, feature_column, threshold, feature_index)
                
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold
                
        return best_feature_index, best_threshold

    def _calculate_information_gain(self, target_vector, feature_column, threshold, feature_index):
        # Calculate entropy before split
        parent_entropy = self._entropy(target_vector)
        
        # Split data based on threshold
        left_indices, right_indices = self._split(feature_column, threshold)
        
        # If split is empty, return zero gain
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        # Calculate weighted entropy after split
        num_samples = len(target_vector)
        num_left, num_right = len(left_indices), len(right_indices)
        entropy_left, entropy_right = self._entropy(target_vector[left_indices]), self._entropy(target_vector[right_indices])
        weighted_entropy = (num_left / num_samples) * entropy_left + (num_right / num_samples) * entropy_right
        
        # Information gain is the difference in entropy
        information_gain = parent_entropy - weighted_entropy
        self.feature_importance_dict[feature_index] += information_gain
        return information_gain
        
    def _split(self, feature_column, threshold):
        left_indices = np.argwhere(feature_column <= threshold).flatten()
        right_indices = np.argwhere(feature_column > threshold).flatten()
        return left_indices, right_indices
        
    def _entropy(self, target_vector):
        label_counts = np.bincount(target_vector)
        probabilities = label_counts / len(target_vector)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])
            
    def _most_common_label(self, target_vector):
        label_counts = Counter(target_vector)
        most_common = label_counts.most_common(1)[0][0]
        return most_common
    
    def _calculate_feature_importances(self):
        total_importance = sum(self.feature_importance_dict.values())
        # Normalize so the importances sum up to 1
        self.feature_importances_ = {feature: importance / total_importance
                                     for feature, importance in self.feature_importance_dict.items()}
        
    
    def predict(self, feature_matrix):
        return np.array([self._traverse_tree(instance, self.root_node) for instance in feature_matrix])
    
    def _traverse_tree(self, instance, node):
        if node.is_leaf_node():
            return node.leaf_value
        
        if instance[node.feature] <= node.threshold:
            return self._traverse_tree(instance, node.left)
        else:
            return self._traverse_tree(instance, node.right)
