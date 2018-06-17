from decision_tree import *
from arff_parser import *

dt = Decision_Tree(30)
dt.make_decision_tree(arff_data("credit_train.arff"))
dt.print_decision_tree()
print(dt.get_test_accurracy(arff_data("credit_test.arff")))