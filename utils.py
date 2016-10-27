from numpy import genfromtxt
from dataset import DataSet
import numpy as np


#####Import data from a file#####
def read_data_file(file_path):
    ##read data from csv file
    data = genfromtxt(file_path, delimiter=',')
    return data


#####Separate the data into train, test and validation datasets#####
##Set the ratio of train, test and validation (ex.train_sample_rate = 0.7, test_sample_rate = 0.2, validation_sample_rate = 0.1).
##If the data_labels matrix is not given, the function will randomly create one.
##The format of the data_features and data_labels is a numpy matrix. The rows should be samples and the columns should be features/labels. 
def sep_data_train_test_val(data_features,train_sample_ratio,test_sample_ratio,validation_sample_ratio,data_labels=None):
    if data_labels is None:
        ##Randomly create a label matrix based on the feature matrix
        data_labels = np.random.randint(1, size=(data_features.shape[0], 10))
        random_index = np.random.randint(10, size=(1,data_features.shape[0]))
        data_labels[np.arange(data_features.shape[0]),random_index]=1

    num_sample_total = data_features.shape[0]
    random_index = np.random.choice(data_features.shape[0], data_features.shape[0], replace=False)

    ##Create index for train, test and validation separately
    train_index = random_index[0:int(num_sample_total*train_sample_ratio)]
    test_index = random_index[int(num_sample_total*train_sample_ratio):int(num_sample_total*train_sample_ratio)+int(num_sample_total*test_sample_ratio)]
    validation_index = random_index[int(num_sample_total*train_sample_ratio)+int(num_sample_total*test_sample_ratio):]

    ## Separate data features into train, test and validation according to the index
    train_features = data_features[train_index,:]
    test_features = data_features[test_index,:]
    validation_features = data_features[validation_index,:]

    ## Separate data labels into train, test adn validation according to the index
    train_labels = data_labels[train_index,:]
    test_labels = data_labels[test_index,:]
    validation_labels = data_labels[validation_index,:]

    ##training data
    train = DataSet(train_features, train_labels,to_one_hot=False)
    ##test data
    test = DataSet(test_features, test_labels,to_one_hot=False)
    ##validation data
    validation = DataSet(validation_features, validation_labels,to_one_hot=False)

    data_set = {'train':train,'test':test,'validation':validation}
    return data_set
