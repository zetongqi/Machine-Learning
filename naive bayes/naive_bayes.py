def count_feature(feature_list, feature_col_num, data):
    counts = [0] * len(feature_list)
    for row in data:
        for i in range(len(feature_list)):
            if row[feature_col_num] == feature_list[i]:
                counts[i] += 1
    return counts

#return feature_counts with the corresponding order as feature2_list, and then feature1_list
def count_two_features(feature1_list, feature1_col_num, feature2_list, feature2_col_num, data):
    feature_counts = []
    for i in range(len(feature2_list)):
        count_list = [0] * len(feature1_list)
        for row in data:
            for j in range(len(feature1_list)):
                if row[feature1_col_num] == feature1_list[j] and row[feature2_col_num] == feature2_list[i]:
                    count_list[j] += 1
        feature_counts.append(count_list)
    return feature_counts

def calculate_prior(label_feature_list, data):
    counts =  count_feature(label_feature_list, -1, data)
    for i in range(len(counts)):
        # Laplace estimates
        counts[i] = counts[i] + 1/(len(data) + len(label_feature_list))
    return counts