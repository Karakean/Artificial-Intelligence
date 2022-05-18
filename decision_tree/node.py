import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    # def find_element_count(self, searched_set, searched_element):
    #     count = 0
    #     for element in searched_set:
    #         if element == searched_element:
    #             count+=1
    #     return count

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0
        # zeros_left = self.left_child.count(0)
        # ones_left = self.left_child.count(1)
        # gini_index_left = 1 - (zeros_left/len(self.left_child))**2-(ones_left/len(self.left_child))**2
        # zeros_right = self.right_child.count(0)
        # ones_right = self.right_child.count(1)
        # gini_index_right = 1 - (zeros_right/len(self.right_child))**2-(ones_left/len(self.right_child))**2
        # total = len(self.left_child) + len(self.right_child)
        # gini_gain = 1 - (len(self.left_child)/total*gini_index_left + len(self.right_child)/total*gini_index_right)
        for split in possible_splits:
            left = y[:split].tolist()
            right = y[split:].tolist()
            zeros_left = left.count(0)
            ones_left = left.count(1)
            if len(left) == 0:
                continue
            gini_index_left = 1 - (zeros_left/len(left))**2-(ones_left/len(left))**2
            zeros_right = right.count(0)
            ones_right = right.count(1)
            if len(right) == 0:
                continue
            gini_index_right = 1 - (zeros_right/len(right))**2-(ones_right/len(right))**2
            gini_gain = 1-(len(left)/len(y))*gini_index_left-(len(right)/len(y))*gini_index_right
            if gini_gain > best_gain:
                best_gain = gini_gain
                best_idx = split
        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y):
        best_gain = -np.inf
        best_split = None

        for d in range(X.shape[1]):
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y):
        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y)
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left)
        self.right_child.train(X_right, y_right)
