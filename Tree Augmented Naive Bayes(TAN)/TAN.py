from arff_parser import *
from naive_bayes import *
import math
import copy


def mutual_information(list1, fea1, col1, list2, fea2, col2, list3, fea3, col3, data):
    cnt = 0
    denom = 0
    fea1_cnt = 0
    fea2_cnt = 0
    for row in data:
        if list1[fea1] == row[col1] and list2[fea2] == row[col2] and list3[fea3] == row[col3]:
            cnt += 1
        if list3[fea3] == row[col3]:
            denom += 1
        if list1[fea1] == row[col1] and list3[fea3] == row[col3]:
            fea1_cnt += 1
        if list2[fea2] == row[col2] and list3[fea3] == row[col3]:
            fea2_cnt += 1
    # p(xi, xj, y)
    a = (cnt+1)/(len(data) + len(list1)*len(list2)*len(list3))
    # p(xi, xj | y)
    b = (cnt+1)/(denom + len(list1)*len(list2))
    # p(xi | y)
    c = (fea1_cnt+1)/(denom + len(list1))
    # p(xj | y)
    d = (fea2_cnt+1)/(denom + len(list2))
    return a*math.log2(b/(c*d))


# returns an adjacency matrix graph
def get_mutual_information(arffdata):
    mutual_info = []
    for i in range(len(arffdata.attributes)):
        temp_list = []
        for j in range(0, len(arffdata.attributes)):
            temp = 0
            if i == j:
                temp_list.append(-1)
                continue
            for y in range(len(arffdata.label.attribute_list)):
                for xi in range(len(arffdata.attributes[i].attribute_list)):
                    for xj in range(len(arffdata.attributes[j].attribute_list)):
                        temp += mutual_information(arffdata.attributes[i].attribute_list, xi, i, arffdata.attributes[j].attribute_list, xj, j, arffdata.label.attribute_list, y, -1, arffdata.data)
            temp_list.append(temp)
        mutual_info.append(temp_list)
    return mutual_info


class Graph:
    def __init__(self):
        self.edges = []
    def add_edge(self, col1, col2):
        self.edges.append([col1, col2])


def find_max_index(m):
    maximum = -100000000
    max_index = [-1, -1]
    for i in range(len(m)):
        for j in range(len(m[i])):
            if m[i][j] > maximum:
                maximum = m[i][j]
                max_index = [i, j]
    return max_index[0], max_index[1]


def find_max_in_row(row):
    maximum = -100000000
    max_index = -1
    for i in range(len(row)):
        if row[i] > maximum:
            maximum = row[i]
            max_index = i
    return max_index, maximum


def prims_algorithm(matrix, arffdata):
    # make a copy
    m = copy.deepcopy(matrix)
    V = list(range(len(arffdata.attributes)))
    Vnew = [0]
    G = Graph()
    
    while set(Vnew) != set(V):
        max_weight = -100000000
        max_x_index = -1
        max_y_index = -1
        for i in Vnew:
            temp_index, temp_weight = find_max_in_row(m[i])
            if temp_weight > max_weight:
                    max_weight = temp_weight
                    max_x_index = i
                    max_y_index = temp_index
        if max_y_index not in Vnew:
            m[max_x_index][max_y_index] = -1
            Vnew.append(max_y_index)
            G.add_edge(max_x_index, max_y_index)
        else:
            m[max_x_index][max_y_index] = -1

    return G


class Node:
    def __init__(self, value):
        self.value = value
        self.parents = []
    def add_parent(self, parent):
        self.append(parent)
    def get_parents_from_graph(self, G):
        for i in G.edges:
            if i[-1] == self.value:
                self.parents.append(i[0])


def TAN_learning(arffdata, G):
    
    # all p(Xi | C, Xparent)
    all_probs = []
    for y in arffdata.label.attribute_list:
        probabilities = []
        for i in range(len(arffdata.attributes)):
            # doesn't include the root
            if i == 0:
                probabilities.append([])
                continue
            n = Node(i)
            n.get_parents_from_graph(G)
            parent = n.parents[0]
            prob_list = []
            #print(arffdata.attributes[i].attribute_list, arffdata.attributes[parent].attribute_list)
            for xi in arffdata.attributes[i].attribute_list:
                temp_list = []
                for xj in arffdata.attributes[parent].attribute_list:
                    cnt = 0
                    cnt2 = 0
                    for row in arffdata.data:
                        if row[i] == xi and row[parent] == xj and row[-1] == y:
                             cnt += 1
                        if row[parent] == xj and row[-1] == y:
                             cnt2 += 1
                    prob = (cnt+1)/(cnt2+len(arffdata.attributes[i].attribute_list))
                    temp_list.append(prob)
                prob_list.append(temp_list)
            probabilities.append(prob_list)
        all_probs.append(probabilities)
        
    # p(Xroot | C)
    y_list = []
    for y in arffdata.label.attribute_list:
        x_list = []
        for x in arffdata.attributes[0].attribute_list:
            pcnt = 0
            ycnt = 0
            for row in arffdata.data:
                if row[0] == x and row[-1] == y:
                    pcnt += 1
                if row[-1] == y:
                    ycnt += 1
            x_list.append((pcnt+1)/(ycnt + len(arffdata.attributes[0].attribute_list)))
        y_list.append(x_list)
            
    return all_probs, y_list


def TAN_inference(row, mutual_prob, root_prob, prior, G, arffdata):
    feature_index = label_to_index(row, arffdata)
    nom_prob0 = 1
    nom_prob1 = 1
    for i in range(1, len(row)-1):
        n = Node(i)
        n.get_parents_from_graph(G)
        parent = n.parents[0]
        nom_prob0 *= mutual_prob[0][i][feature_index[i]][feature_index[parent]]
        nom_prob1 *= mutual_prob[1][i][feature_index[i]][feature_index[parent]]
    nom_prob0 *= root_prob[0][feature_index[0]]*prior[0]
    nom_prob1 *= root_prob[1][feature_index[0]]*prior[1]
    
    prob1 = nom_prob0/(nom_prob0 + nom_prob1)
    prob2 = nom_prob1/(nom_prob0 + nom_prob1)
    return prob1, prob2


class TAN:
	def __init__(self):
		self.arffdata = None
		self.G = None
		self.mutual_prob = None # p(Xi | Class, Xparent)
		self.root_prob = None # p(Xroot | Class)
		self.prior = None # p(Class)

	def TAN_train(self, arffdata):
		self.arffdata = arffdata
		mutual_info = get_mutual_information(arffdata)
		self.G = prims_algorithm(mutual_info, self.arffdata)
		self.mutual_prob, self.root_prob = TAN_learning(arffdata, self.G)
		self.prior = calculate_prior(self.arffdata.label.attribute_list, self.arffdata.data)

	def TAN_Classify(self, row):
		prob1, prob2 = TAN_inference(row, self.mutual_prob, self.root_prob, self.prior, self.G, self.arffdata)
		if prob1 >= prob2:
			return self.arffdata.label.attribute_list[0]
		if prob1 <= prob2:
			return self.arffdata.label.attribute_list[1]

	def get_test_accuracy(self, test_arffdata):
		correct_cnt = 0
		for row in test_arffdata.data:
			if self.TAN_Classify(row) == row[-1]:
				correct_cnt += 1
		return correct_cnt/len(test_arffdata.data)










