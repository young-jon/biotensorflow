from numpy import genfromtxt
from dataset import DataSet
import numpy as np


def read_data_file(file_path):
    '''
    Function 'read_data_file' is to import data from a csv file.
    file_path: the path of a csv file
    '''
    
    ##read data from csv file
    data = genfromtxt(file_path, delimiter=',')
    return data


def sep_data_train_test_val(data_features,train_sample_ratio,test_sample_ratio,validation_sample_ratio,data_labels=None):
    '''
    Function 'sep_data_train_test_val' is to separate the data into train, test and validation datasets based
    on the ratio of train, test and validation (ex.train_sample_rate = 0.7, test_sample_rate = 0.2, validation_sample_rate = 0.1).
    data_features: a numpy matrix. The rows should be samples and the columns should be features
    train_sample_ratio: the ratio of the training samples (a constant between 0 and 1)
    test_sample_ratio: the ratio of the test samples (a constant between 0 and 1)
    validation_sample_ratio: the ratio of the validation samples (a constant between 0 and 1)
    data_labels: a numpy matrix. The rows should be samples and the columns should be labels. If the data_labels matrix is not given, the function will randomly create one.
    '''
    
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
