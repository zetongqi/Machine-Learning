from arff_parser import *


def count_feature(feature_list, feature_col_num, data):
    counts = [0] * len(feature_list)
    for row in data:
        for i in range(len(feature_list)):
            if row[feature_col_num] == feature_list[i]:
                counts[i] += 1
    return counts


def count_joint_probability(feature1_list, feature1_col_num, feature2_list, feature2_col_num, data):
    label_counts = count_feature(feature2_list, -1, data)
    feature_counts = []
    for i in range(len(feature2_list)):
        count_list = [0] * len(feature1_list)
        for row in data:
            for j in range(len(feature1_list)):
                if row[feature1_col_num] == feature1_list[j] and row[feature2_col_num] == feature2_list[i]:
                    count_list[j] += 1
        for k in range(len(count_list)):
            count_list[k] = (count_list[k]+1) / (label_counts[i] + len(feature1_list))
        feature_counts.append(count_list)
    return feature_counts


def calculate_prior(label_feature_list, data):
    counts =  count_feature(label_feature_list, -1, data)
    for i in range(len(counts)):
        # Laplace estimates
        counts[i] = (counts[i] + 1)/(len(data) + len(label_feature_list))
    return counts


def get_all_joint_probability(arffdata):
    probabilities = []
    for i in range(len(arffdata.attributes)):
        probabilities.append(count_joint_probability(arffdata.attributes[i].attribute_list, i, arffdata.label.attribute_list, -1, arffdata.data))
    return probabilities

def label_to_index(row, arffdata):
    feature_index = []
    for i in range(len(row)):
        for j in range(len(arffdata.all_attributes[i].attribute_list)):
            if row[i] == arffdata.all_attributes[i].attribute_list[j]:
                feature_index.append(j)
                continue
    return feature_index


def naive_bayes_learning(arffdata):
    joint_probabilities = get_all_joint_probability(arffdata)
    prior = calculate_prior(arffdata.label.attribute_list, arffdata.data)
    return joint_probabilities, prior


def naive_bayes_inference(row, train_arffdata, joint_probabilities, prior):
    feature_index = label_to_index(row, train_arffdata)
    nominator1 = 1
    nominator2 = 1
    for i in range(len(feature_index)-1):
        nominator1 = nominator1 * joint_probabilities[i][0][feature_index[i]]
        nominator2 = nominator2 * joint_probabilities[i][1][feature_index[i]]
    nominator1 = nominator1 * prior[0]
    nominator2 = nominator2 * prior[1]
    denominator = nominator1 + nominator2
    if nominator1/denominator >= nominator2/denominator:
        return 0
    if nominator1/denominator < nominator2/denominator:
        return 1


class naive_bayes:
    def __init__(self):
        self.arffdata = None
        self.joint_probabilities = None
        self.prior = None

    def learn(self, trainfile):
        self.arffdata = arff_data(trainfile)
        self.joint_probabilities, self.prior = naive_bayes_learning(self.arffdata)

    def classify(self, row):
        return self.arffdata.label.attribute_list[naive_bayes_inference(row, self.arffdata, self.joint_probabilities, self.prior)]

    def classify_test_set(self, test_data):
        correct_cnt = 0
        for row in test_data:
            result = self.classify(row)
            if result == row[-1]:
                correct_cnt += 1
            print(result)
        print("Correctly classified: ", correct_cnt)
        print("Accuracy: ", correct_cnt/len(test_data))
