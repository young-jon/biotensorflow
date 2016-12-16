from numpy import genfromtxt
from dataset import DataSet
import numpy as np


def read_data_file(file_path):
    '''
    This function 'read_data_file' is to import data from a csv file.

    Args:
    file_path: the path of a csv file which is passed as a string.

    Returns:
    A numpy matrix

    Example Usage:
    file = read_data_file('/Users/luc17/Desktop/PDX project/pdx_bimodal_binary_feature_selected.csv')
    '''
    
    ##read data from csv file
    data = genfromtxt(file_path, delimiter=',')
    return data


def sep_data_train_test_val(data_features,train_sample_ratio,test_sample_ratio,validation_sample_ratio,data_labels=None):
    '''
    This function 'sep_data_train_test_val' is to separate the data into training, test and validation datasets based
    on the ratio of training, test and validation (ex.train_sample_rate = 0.7, test_sample_rate = 0.2, validation_sample_rate = 0.1).

    Args:
    data_features: a numpy matrix. The rows should be samples and the columns should be features.
    train_sample_ratio: a number between 0 and 1 which represents the ratio of the training samples. The sum of train_sample_ration, test_sample_ratio and validation_sample_ration should be 1. 
    test_sample_ratio: a number between 0 and 1 which represents the ratio of the test samples.
    validation_sample_ratio: a number between 0 and 1 which represents the ratio of the validation samples.
    data_labels: a numpy matrix. The rows should be samples and the columns should be labels. If the data_labels matrix is not given, the function will randomly create one.

    Returns:
    A numpy dictionary containing separated training, test and validation dataset with keys 'train', 'test' and 'validation'.

    Example usage:
    dataset = sep_data_train_test_val(data_features,0.7,0.2,0.1)
    training_dataset = dataset['train']
    test_dataset = dataset['test']
    validation_dataset = dataset['validation']
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

### IMAGING UTILS
def get_image_dims(n_input):
    ''' converts 1-D integer to 2-D representation for plotting as an image

    Args
    n_input (int):  number of features in the data that you want to plot as an image

    Returns
    A 2-D tuple (of dimensions) for plotting

    Usage
    print(get_image_dims(784))
    # (28, 28)
    print(get_image_dims(500))
    # (25, 20)

    '''
    ### check for perfect square
    if not (np.sqrt(n_input) - int(np.sqrt(n_input))):
        dimensions = (int(np.sqrt(n_input)), int(np.sqrt(n_input)))
    ### if not perfect square
    else:
        dim1 =[]
        dim2=[]
        mid = int(np.floor(np.sqrt(n_input)))
        for i in range(mid):
            if (n_input % (i+1)) == 0:
                # print(i+1)
                dim1.append(i+1)
        for i in range(mid,n_input):
            if (n_input % (i+1)) == 0:
                # print(i+1)
                dim2.append(i+1)
        dimensions = (min(dim2), max(dim1))
        if 1 in dimensions:
            print('prime number of features')
    return dimensions

