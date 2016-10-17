import numpy as np
from numpy import genfromtxt

##1a and 1b. randomly create a binary/real valued numpy array (~5000*500(samples*features))
##Choose one of the followings based on your needs
##Create binary numpy array
data_features = random_create_data_features("binary",5000, 500)
"""
##Create real numpy array
data_features = random_create_data_features("real", 5000, 500)
##Import data from your file. Please change the file_path based on your own. The file should only contain the numeric data (either binary or real valued number).
file_path_local = '/Users/luc17/Desktop/PDX project/pdx_bimodal_binary_feature_selected.csv'
data_features = random_create_data_features("self_defined",file_path=file_path_local)
"""

##Randomly create a corresponding hot vector numpy array(~5000*10(samples*labels)) 

data_labels = np.random.randint(1, size=(data_features.shape[0], 10))
random_index = np.random.randint(10, size=(1,data_features.shape[0]))
data_labels[np.arange(data_features.shape[0]),random_index]=1

##Separate the data into train, test and validation
##Set the percentage of train, test and validation (train:70%, test:20%, validation:10%)

num_sample_total = data_features.shape[0]
train_sample_rate = 0.7
test_sample_rate = 0.2
validation_sample_rate = 0.1
random_index = np.random.choice(data_features.shape[0], data_features.shape[0], replace=False)

##Create index for train, test and validation separately

train_index = random_index[0:int(num_sample_total*train_sample_rate)]
test_index = random_index[int(num_sample_total*train_sample_rate):int(num_sample_total*train_sample_rate)+int(num_sample_total*test_sample_rate)]
validation_index = random_index[int(num_sample_total*train_sample_rate)+int(num_sample_total*test_sample_rate):]

## Separate data features into train, test and validation according to the index

train_features = data_features[train_index,:]
test_features = data_features[test_index,:]
validation_features = data_features[validation_index,:]

## Separate data labels into train, test adn validation according to the index

train_labels = data_labels[train_index,:]
test_labels = data_labels[test_index,:]
validation_labels = data_labels[validation_index,:]

##Create a new dataset to save the train, test and validation data together
data_set = DataSets()

###save training data
data_set.train = DataSet(train_features, train_labels,to_one_hot=False)
###save test data
data_set.test = DataSet(test_features, test_labels,to_one_hot=False)
###save validation data
data_set.validation = DataSet(validation_features, validation_labels,to_one_hot=False)

##use data_set as input for dnn.py, the only code needs to be changed in the dnn.py is data = data_set


##define a function to create data randomly or import from a file
def random_create_data_features(type,num_samples="",num_features="",file_path=""):
    if type == "binary":
        data_feature = np.random.randint(2, size=(num_samples, num_features))
    if type == "real":
        data_feature = np.random.uniform(0,1,size = (num_samples,num_features))
    if type == "self_defined":
        ##read from local .csv file
        data_feature = genfromtxt(file_path, delimiter=',')
    return data_feature


class DataSet:
    """
    Defines a ML DataSet

    Arguments:
    features (numpy.array): NumPy array of 'm' example features
    labels (numpy.array): NumPy array of 'm' example labels
    flatten (bool): Whether or not to flatten features into [m x n] array
    to_one_hot (bool): Whether or not to convert labels into array of one-hot vectors

    Usage:
    import numpy as np

    # Create random array of 10, 28x28x1 "images"
    features = np.random.rand(10, 28, 28, 1)

    # Create labels array of image classes (0 = car, 1 = person, 2 = tree)
    labels = np.array([0, 1, 2, 1, 2, 1, 0, 0, 0, 1])

    # Create data set of images and convert image labels to one-hot vectors
    image_dataset = DataSet(features, labels, to_one_hot=True)

    # Get next batch of 5 images
    (batch_features, batch_labels) = image_dataset.next_batch(5)

    """
    def __init__(self, features, labels, flatten=False, to_one_hot=False):
        assert(features.shape[0] == labels.shape[0])
        self._features = features if not flatten else features.reshape((features.shape[0], -1), order="F")
        self._labels = labels if not to_one_hot else self.to_one_hot(np.squeeze(labels).astype(int))

        self._num_examples = features.shape[0]
        self._epoch_count = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def to_one_hot(self, vector):
        """ Converts vector of values into one-hot array """

        # Get number of classes from max value in vector
        num_classes = np.max(vector) + 1

        # Create array of zeros
        result = np.zeros(shape=(vector.shape[0], num_classes))

        # Set appropriate values to 1
        result[np.arange(vector.shape[0]), vector] = 1

        # Return as integer NumPy array
        return result.astype(int)

    def next_batch(self, batch_size):
        """
        Gets next batch of examples given a batch size

        Adapted from TF MNIST DataSet class code
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished 1 epoch
            self._epoch_count += 1

            # Shuffle dataset for next epoch
            permutation = np.arange(self._num_examples)
            np.random.shuffle(permutation)
            self._features = self._features[permutation]
            self._labels = self._labels[permutation]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return (self._features[start:end], self._labels[start:end])
    
class DataSets(DataSet):
    def __init__(self):
        pass
