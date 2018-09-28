from numpy import genfromtxt
from dataset import DataSet
import numpy as np
import tensorflow as tf
import itertools
import daft
import matplotlib.pyplot as plt


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


### INITIALIZATION

### Goodfellow book recommends treating weight initialization as a hyperparameter.
### E.g. could try xavier_init, xavier_init2, tf.truncated_normal_initializer(stddev=stddev)) stdev=0.02 or other,
### tf.truncated_normal(shape, stddev=0.1). truncated_normal_initializer used by Improved_gan and Info_Gan code.
### For biases could use tf.constant(0.1, shape=shape) --- from tensorflow tutorial, zeros, or random_normal
def xavier_init(size):
    print('Using Xavier initialization')
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def xavier_init2(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)



### OBJECTIVE FUNCTIONS
def rmse(logits, labels):
    return tf.reduce_mean(tf.pow(labels - logits, 2))




### VISUALIZE SIM DATA/WEIGHTS AS GRAPHS
### DRAW GRAPH ###
def plot_hierarchy(all_matrices, arch, asp=1, nu=0.4, gu=4):
    '''
    arch: list of hidden layers and DEGs
    all_matrices: list of numpy ndarray matrices for arch above
    '''
    pgm = daft.PGM([np.max(arch),len(arch)], origin=[.5, .5], node_unit=nu, grid_unit=gu,
            directed=True)
    # pgm = daft.PGM([arch[-1]+1,len(arch)+1]) #bigger
    # Add nodes to plot
    i=1
    for yi, layer in enumerate(arch):
        for xi in range(layer):
            pgm.add_node(daft.Node(str(i), "", xi+ ((np.max(arch)-layer)/2) +1, yi+1, aspect=asp)) #weird code to center layers
            #use aspect to shrink or stretch nodes. can maybe get up to 300 visible nodes on a screen using aspect = 10
            #The aspect ratio width/height
            # print i, xi+1, yi +1
            i+=1
    # Enumerate all edges in Daft's required syntax 
    # (i.e. str((4,7)) represents edge from node 4 to node 7)
    # all edges where assigned to a number (increasing starting at 1) in the previous step in order to identify them in 
    # this step. this is why we have to use i_buffer and j_buffer. because all matrices start numbering at rows and col at 0.
    all_edges=[]
    i_buffer = 0
    j_buffer = arch[0]
    for mi, matrix in enumerate(all_matrices):
        for i, row in enumerate(matrix):
            for j, possible_edge in enumerate(row):
                if possible_edge == 1:
                    all_edges.append((i+i_buffer+1,j+j_buffer+1))
        i_buffer += arch[mi]  #or j_buffer
        j_buffer += arch[mi+1]
    # Add edges to graph
    for e in all_edges:
        pgm.add_edge(str(e[0]), str(e[1]))
    # Display and save image
    pgm.render()
    # pgm.figure.savefig(file_name)
    # pgm.figure.savefig(file_name, dpi=200)
    # plt.show()

### DRAW SGA GRAPH ###
def plot_sga_loc(bin_sga_edge_matrix, arch, num_SGAs, asp=1, nu=0.4):
    SGA_graph_arch = [num_SGAs]+arch[:-1]
    pgm = daft.PGM([np.max(SGA_graph_arch),len(SGA_graph_arch)], origin=[.5, .5], node_unit=nu, grid_unit=8,
            directed=True)

    # Add nodes to plot for SGA graph
    i=1 
    for yi, layer in enumerate(SGA_graph_arch):
        for xi in range(layer):
            pgm.add_node(daft.Node(str(i), "", xi+ ((np.max(SGA_graph_arch)-layer)/2) +1, yi+1, aspect=asp)) #weird code to center layers
            #use aspect to shrink or stretch nodes. can maybe get up to 300 visible nodes on a screen using aspect = 10
            #The aspect ratio width/height
            # print i, xi+1, yi+1
            i+=1

    ### ADD SGA EDGES
    all_edges=[]

    j_buffer = SGA_graph_arch[0]
    for i, row in enumerate(bin_sga_edge_matrix):
        for j, possible_edge in enumerate(row):
            if possible_edge == 1:
                all_edges.append((i+1,j+j_buffer+1))

    # Add edges to graph
    for e in all_edges:
        pgm.add_edge(str(e[0]), str(e[1]))

    pgm.render()



def forward_prop(sga,w,b,activation,output_activation):
    '''forward propagation through a RINN. returns output activations. This assumes output is binary 
    (ie using sigmoid output function).
    sga = (numpy ndarray) input vector
    w = list of  weight matrices (each matrix is numpy ndarray) (8 hidden layer network)
    b = list of bias vectors (each vector is numpy ndarray)
    activation = activation function for hidden layers (a python function)
    output_activation = activation function for output layer (e.g. sigmoid -- for binary, or identity -- for regression)
    '''
    # from scipy.special import expit  #sigmoid/logistic function
    
    da = np.asarray(sga)     
    h1 = activation(np.dot(da, w[0]) + b[0])
    h2 = activation(np.dot(np.concatenate((h1, da)), w[1]) + b[1])
    h3 = activation(np.dot(np.concatenate((h2, da)), w[2]) + b[2])
    h4 = activation(np.dot(np.concatenate((h3, da)), w[3]) + b[3])
    h5 = activation(np.dot(np.concatenate((h4, da)), w[4]) + b[4])
    h6 = activation(np.dot(np.concatenate((h5, da)), w[5]) + b[5])
    h7 = activation(np.dot(np.concatenate((h6, da)), w[6]) + b[6])
    h8 = activation(np.dot(np.concatenate((h7, da)), w[7]) + b[7])
    deg = output_activation(np.dot(h8, w[8]) + b[8])

    return deg

def softplus(x):
    return np.log(1.0 + np.exp(x))

def nprelu(x):
    return x * (x > 0)


def get_sparsity(weights, threshold): #weights is list of tensorflow variables 
    all_matrices = []
    for i in weights:
        all_matrices.append(np.where(np.abs(i.eval()) > threshold, 1, 0))
    active_edges = 0
    possible_edges = 0
    for j in all_matrices:
        active_edges += np.sum(j)
        possible_edges += (j.shape[0]*j.shape[1])
    return active_edges, active_edges/possible_edges
