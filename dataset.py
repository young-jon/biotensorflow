import numpy as np

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
