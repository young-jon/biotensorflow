import numpy as np

class DataSet:
    def __init__(self, features, labels, flatten=True, to_one_hot=False):
        """
        Construct a DataSet given NumPy arrays of features and labels
        """

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