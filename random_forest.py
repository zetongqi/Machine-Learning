from arff_parser import *
from decision_tree import *
import copy

class Random_Forest:
    def __init__(self, arffdata):
        self.num_of_trees = 0
        self.arffdata = arffdata
        self.tree_list = []
        self.arffdata_list = []

    def grow_forest(self, num_of_trees = 7):
        self.num_of_trees = num_of_trees
        for i in range(num_of_trees):
            temp = arff_data(self.arffdata.filename)
            temp.mutate(0.7, 0.5)
            self.arffdata_list.append(temp)
        
        for i in range(self.num_of_trees):
            dt = Decision_Tree(30)
            dt.make_decision_tree(self.arffdata_list[i])
            self.tree_list.append(dt)

    def get_majority(self, array):
        unique_list = list(set(array))
        occurrences = [0] * len(unique_list)

        for i in array:
            for j in range(len(unique_list)):
                if i == unique_list[j]:
                    occurrences[j] += 1
        
        largest = -10000000
        largest_index = -1
        for i in range(len(occurrences)):
            if occurrences[i] > largest:
                largest = occurrences[i]
                largest_index = i
        
        return unique_list[largest_index]

    def classify(self, test_arff, rownum):
        results = []
        for i in range(self.num_of_trees):
            temp_test_arff = arff_data(test_arff.filename)
            temp_test_arff.mutate_with_num_list(self.arffdata_list[i].mutate_num_list)
            a = self.tree_list[i].classify(self.arffdata_list[i], self.tree_list[i].tree, temp_test_arff.data[rownum])
            results.append(a)
        return self.get_majority(results)

    def classification_accuracy(self, test_arff):
        cnt = 0
        for i in range(len(test_arff.data)):
            if self.classify(test_arff, i) == test_arff.data[i][-1]:
                cnt += 1
        return cnt/len(test_arff.data)