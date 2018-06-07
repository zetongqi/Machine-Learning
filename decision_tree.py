from arff_parser import *
import sys
import math

class Question:
    def __init__(self, arffdata, feature_num, value, less_than = 1):
        #feature is the column number in the data set, nominal=1: feature is nominal
        self.arffdata = arffdata
        self.feature = arffdata.attributes[feature_num].name
        self.feature_num = feature_num
        self.value = value
        self.type = arffdata.all_attributes[feature_num].type
        self.less_than = less_than
    
    def is_numeric(self):
        if self.arffdata.all_attributes[feature_num].type == "real":
            return 1
        if self.arffdata.all_attributes[feature_num].type == "nominal":
            return 0
        
    def match(self, row):
        if self.type == "nominal":
            if row[self.feature_num] == self.value:
                return True
            else:
                return False
        if self.type == "real":
            if self.less_than == 0:
                if row[self.feature_num] > self.value:
                    return True
                else:
                    return False
            if self.less_than == 1:
                if row[self.feature_num] <= self.value:
                    return True
                else:
                    return False
        
    def __repr__(self):
        if self.arffdata.all_attributes[self.feature_num].type == "nominal":
            return "%s == %s" % (self.feature, self.value)
        if self.arffdata.all_attributes[self.feature_num].type == "real":
            if self.less_than == 0:
                return "%s > %s" % (self.feature, self.value)
            if self.less_than == 1:
                return "%s <= %s" % (self.feature, self.value)

class Node:
    def __init__(self, questions, data):
        self.questions = questions
        self.data = data
        self.children = []
        
    def add_children(self, child):
        if isinstance(child, Node):
            self.children.append(child)
        else:
            print("Error: Children must be of Node type")
    """def __repr__(self):
        if self.question.is_numeric() == 1:
            return "%s == %s ?" % (self.question.feature, self.question.value)
        if self.question.is_numeric() == 0:
            return "%s > %s ?" % (self.question.feature, self.question.value)"""

class Leaf:
    def __init__(self, data):
        self.data = data

def get_feature_data(column_num, data):
    column = []
    for row in data:
        column.append(row[column_num])
    return column

#given the arff data and feature number, return a list of candidate splits
#arffdata carries feature info and data carries actual data
def determine_candidate_split(arffdata, data, feature_num):
    feature_value_list = get_feature_data(feature_num, data)
    
    if arffdata.all_attributes[feature_num].type == "real":
        feature_value_list = sorted(feature_value_list)
        candidates = []
        for i in range(len(feature_value_list)-1):        
            candidates.append((feature_value_list[i]+feature_value_list[i+1])/2)
        if len(sorted(list(set(candidates)))) == 0:
            return [-10000000]
        else:
            return sorted(list(set(candidates)))
    
    if arffdata.all_attributes[feature_num].type == "nominal":
        candidates = arffdata.all_attributes[feature_num].attribute_list
        return candidates

def split_data(data, arffdata, feature_num, threshold):
    attributes = arffdata.all_attributes[feature_num].attribute_list
    if arffdata.all_attributes[feature_num].type == "real":
        nominal = 0
    if arffdata.all_attributes[feature_num].type == "nominal":
        nominal = 1
    splited_data = []
    
    if nominal == 1:
        for i in range(len(attributes)):
            split = []
            for j in data:
                if j[feature_num] == attributes[i]:
                    split.append(j)
            splited_data.append(split)
    
    if nominal == 0:
        left = []
        right = []
        for i in data:
            if i[feature_num] > threshold:
                left.append(i)
            else:
                right.append(i)
        splited_data.append(right)
        splited_data.append(left)
        
    return splited_data

def label_entropy(data, arffdata):
    cnt = 0
    for i in data:
        if arffdata.label.attribute_list[0] == i[-1]:
            cnt += 1
    if len(data) == 0:
        return 0
    p = cnt/len(data)
    return -p*math.log2(p)-(1-p)*math.log2(1-p)

def Entropy(p):
    return -p*math.log2(p)

def real_feature_entropy(data, arffdata, feature_num, threshold):
    if len(data) == 0:
        return 0
    cnt_greater = 0
    cnt_pos_greater = 0
    cnt_pos_less = 0
    for i in data:
        if i[feature_num] > threshold:
            cnt_greater += 1
        if i[-1] == arffdata.label.attribute_list[0] == i[-1] and i[feature_num] > threshold:
            cnt_pos_greater += 1
        if i[-1] == arffdata.label.attribute_list[0] == i[-1] and i[feature_num] <= threshold:
            cnt_pos_less += 1
    p_greater = cnt_greater/len(data)
    if cnt_greater == 0:
        p_cnt_pos_greater = 0
    else:
        p_cnt_pos_greater = cnt_pos_greater/cnt_greater
    p_cnt_pos_less = cnt_pos_less/(len(data)-cnt_greater)
    if p_cnt_pos_greater != 0:
        E1 = Entropy(p_cnt_pos_greater)
    else:
        E1 = 0
    if 1-p_cnt_pos_greater != 0:
        E2 = Entropy(1-p_cnt_pos_greater)
    else:
        E2 = 0
    if p_cnt_pos_less != 0:
        E3 = Entropy(p_cnt_pos_less)
    else:
        E3 = 0
    if 1-p_cnt_pos_less:
        E4 = Entropy(1-p_cnt_pos_less)
    else:
        E4 = 0
    entropy = p_greater*(E1+E2) + (1-p_greater)*(E3+E4)
    return entropy

def nominal_feature_entropy(data, arffdata, feature_num):
    if len(data) == 0:
        return 0
    entropy = 0
    for i in arffdata.all_attributes[feature_num].attribute_list:
        cnt = 0
        cnt_pos = 0   
        for row in data:
            if row[feature_num] == i:
                cnt += 1
            if row[feature_num] == i and row[-1] == arffdata.label.attribute_list[0]:
                cnt_pos += 1
        if cnt == 0:
            p_cnt_pos = 0
        else:
            p_cnt_pos = cnt_pos/cnt
        if p_cnt_pos != 0:
            E1 = Entropy(p_cnt_pos)
        else:
            E1 = 0
        if 1- p_cnt_pos != 0:
            E2 = Entropy(1-p_cnt_pos)
        else:
            E2 = 0
        entropy += (cnt/len(data))*(E1+E2)
    return entropy

def entropy(data, arffdata, feature_num, threshold):
    if arffdata.all_attributes[feature_num].type == "real":
        return real_feature_entropy(data, arffdata, feature_num, threshold)
    if arffdata.all_attributes[feature_num].type == "nominal":
        return nominal_feature_entropy(data, arffdata, feature_num)

def info_gain(data, arffdata, feature_num, threshold):
    return label_entropy(data, arffdata)-entropy(data, arffdata, feature_num, threshold)

def find_best_numeric_candidate(data, arffdata, feature_num):
    candidates = determine_candidate_split(arffdata, data, feature_num)
    print("here new", get_feature_data(0, data))
    best_candidate = candidates[0]
    best_info_gain = info_gain(data, arffdata, feature_num, candidates[0])
    
    for i in range(1, len(candidates)):
        if info_gain(data, arffdata, feature_num, candidates[i]) > best_info_gain:
            best_info_gain = info_gain(data, arffdata, feature_num, candidates[i])
            best_candidate = candidates[i]
    return best_candidate, best_info_gain

def if_same_class(data):
    if len(list(set(get_feature_data(-1, data)))) == 1:
        return 1
    else:
        return 0

def find_best_split(data, arffdata):
    best_split = 0
    best_info_gain = -10000000
    
    for i in range(len(arffdata.attributes)):
        if arffdata.attributes[i].type == "real":
            numeric, _ = find_best_numeric_candidate(data, arffdata, i)
            gain = info_gain(data, arffdata, i, numeric)
            if gain > best_info_gain:
                best_info_gain = gain
                best_split = i
        if arffdata.attributes[i].type == "nominal":
            gain = info_gain(data, arffdata, i, 0)
            if gain > best_info_gain:
                best_info_gain = gain
                best_split = i
    return best_split, best_info_gain

def get_majority(data, arffdata):
    cnt1 = 0
    cnt2 = 0
    for i in data:
        if i[-1] == arffdata.label.attribute_list[0]:
            cnt1 += 1
        if i[-1] == arffdata.label.attribute_list[1]:
            cnt2 += 1
    if cnt1 >= cnt2:
        return arffdata.label.attribute_list[0]
    else:
        return arffdata.label.attribute_list[1]

class Decision_Tree:
    #stopping building tree if number of instances is less than m
    def __init__(self, m):
        self.m = m
        self.tree = None
        self.arffdata = None
    
    def make_decision_tree(self, arffdata):
        self.arffdata = arffdata
        self.tree = self.build_tree(arffdata.data, arffdata, self.m)

    def build_tree(self, subdata, arffdata, m): 
        if if_same_class(subdata) == 1:
            return Leaf(subdata)
        best_split, best_info_gain = find_best_split(subdata, arffdata)
        if len(subdata) < m or best_info_gain <= 0:
            return Leaf(subdata)
        
        else:
            #best_split, best_info_gain = find_best_split(subdata, arffdata)
            #print(best_split)
            if arffdata.attributes[best_split].type == "real":
                threshold, _ = find_best_numeric_candidate(subdata, arffdata, best_split)
                splited_data = split_data(subdata, arffdata, best_split, threshold)
                questions = [Question(arffdata, best_split, threshold), Question(arffdata, best_split, threshold, 0)]
                node = Node(questions, subdata)
                for i in splited_data:
                    node.children.append(self.build_tree(i, arffdata, m))
                return node
            
            if arffdata.attributes[best_split].type == "nominal":
                splited_data = split_data(subdata, arffdata, best_split, 0)
                questions = []
                for i in arffdata.attributes[best_split].attribute_list:
                    questions.append(Question(arffdata, best_split, i))
                node = Node(questions, subdata)
                for i in splited_data:
                    node.children.append(self.build_tree(i, arffdata, m))
                return node

    def print_decision_tree(self):
        if self.tree is None:
            print("Please make tree before print!")
        else:
            self.print_tree(self.arffdata, self.tree)

    def print_tree(self, arffdata, node, spacing=""):

        if isinstance(node, Leaf):
            return

        for i in range(len(node.children)):
            if type(node.children[i]) is Node:
                print(spacing + str(node.questions[i]), sep = "")
            else:
                print(spacing + str(node.questions[i])+" ("+get_majority(node.children[i].data, arffdata)+")", sep = "")
            self.print_tree(arffdata, node.children[i], spacing + "|   ")

    def classify(self, arffdata, node, row):
        if isinstance(node, Leaf):
            return get_majority(node.data, arffdata)

        for i in range(0, len(node.questions)):
            if node.questions[i].match(row):
                return self.classify(arffdata, node.children[i], row)
            else:
                continue

    def get_test_accurracy(self, test_arffdata):
        return self.get_classification_accuracy(test_arffdata, self.tree)

    def get_classification_accuracy(self, test_arffdata, node):
        cnt = 0
        for i in test_arffdata.data:
            if self.classify(test_arffdata, node, i) == i[-1]:
                cnt += 1
        print(cnt, len(test_arffdata.data))
        return cnt/len(test_arffdata.data)