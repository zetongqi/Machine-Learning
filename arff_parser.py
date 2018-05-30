import sys

#a feature class contains class type(real or nominal) and attribute list
class feature:
    def __init__(self, name, Type):
        self.name = name
        if Type == 0:
            self.type = "real"
        else:
            self.type = "nominal"
        self.attribute_list = []

    def set_attribute(self, attribute_list):
        self.attribute_list = attribute_list

    def print_attribute(self):
        print(self.type, " ", self.attribute_list)

    def get_attribute(self):
        return self.attribute_list

    def feature_type(self):
        return self.type
    
    def __repr__(self):
        return " %s(%s): %s" % (self.name, self.type, self.attribute_list)


class arff_data:
    def __init__(self, file_name):
        data_file = open(file_name, "r")
        data = []
        num_attributes = 0
        attributes = []

        for line in data_file:

            if line.startswith("@attribute"):

                num_attributes += 1
                
                startIndex = line.find('\'')
                if startIndex != -1: #i.e. if the first quote was found
                    endIndex = line.find('\'', startIndex + 1)
                    if startIndex != -1 and endIndex != -1: #i.e. both quotes were found
                        feature_name = line[startIndex+1:endIndex]

                if "{" in line and "}" in line:
                    word = line[line.index("{") + 1 : line.index("}")]
                    word = word.rstrip("\n\r")
                    word = word.strip()                
                    attribute_list = word.split(",")                 
                    temp = feature(feature_name, 1)
                    temp.set_attribute(attribute_list)
                    attributes.append(temp)
                else:
                    # attributes is a list of type "feature"
                    attributes.append(feature(feature_name, 0))

            if line.startswith("@"):
                continue

            line = line.rstrip("\n\r")
            my_list = line.split(",")
            data.append(my_list)

        # a list to indicate if the feature in the coressponding position is real=1 or nominal=0
        real_list = []
        for i in attributes:
            if i.type == "real":
                real_list.append(1)
            if i.type == "nominal":
                real_list.append(0)
        
        for j in range(len(data[0])):
            if real_list[j] == 1:
                for i in range(len(data)):
                    data[i][j] = float(data[i][j])

        self.file_name = data_file
        self.data = data
        self.num_attributes = num_attributes-1
        self.label = attributes[-1]
        self.attributes = attributes[:-1]
        self.m = len(self.data)
        self.n = len(self.data[0])
    
    def print_attributes(self):
        for i in self.attributes:
            i.print_attribute()

    def get_attributes(self):
        return self.attributes
    
    def get_num_attributes(self):
        return self.num_attributes

    def get_labels(self):
        return self.label
    
    def print_labels(self):
        self.label.print_attribute()

    def get_data(self):
        return self.data

    #get the entire column of feature
    def get_feature_data(self, column_num):
        column = []
        for row in self.data:
            column.append(row[column_num])
        return column