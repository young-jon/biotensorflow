from __future__ import (
    absolute_import, division, print_function, unicode_literals
)

import numpy as np
import tensorflow as tf
import logging
import sys


class GridSearch:
    """
    Class for optimizing TF model structure/hyperparameters with validation data

    Example Usage:
    # train_dataset = DataSet(...)
    # valid_dataset = DataSet(...)

    conv1_filter_width = [12, 16]
    conv1_num_feature_maps = [16, 24]

    conv2_filter_width = [2, 4]
    conv2_num_feature_maps = [4, 8]
    maxpool2_size = [2, 4]

    fc1_size = [1, 8]

    # Generate "grid" of hyperparameters
    grid = list(itertools.product(conv1_filter_width, conv1_num_feature_maps,
                                  conv2_filter_width, conv2_num_feature_maps,
                                  maxpool2_size, fc1_size))

    # Setup model configurations
    model_configs = []
    for hyperparams in grid:
        c1_fw, c1_k, c2_fw, c2_k, mp2_w, fc1_n = hyperparams

        # Create model directory
        model_dir = (base_path + "/saved_models_worker_{0}/{1}/{2}" \
                     .format(worker_id, factor_name, model_name))

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            os.makedirs(model_dir + "/checkpoints")
            os.makedirs(model_dir + "/tensorboard")
            logger.debug("Created model directory: {0}".format(model_dir))

        # CNN structure
        structure_def = ["INPUT:4,350,1",
                         "CONV:4,{0},1,{1}:1:relu".format(c1_fw, c1_k),
                         "CONV:1,{0},{1},{2}:1:relu".format(c2_fw, c1_k, c2_k),
                         "MAXPOOL:1,{0}:2".format(mp2_w),
                         "FC:{0}:relu".format(fc1_n),
                         "OUTPUT:2:softmax"]

        # CNN config
        model_config = {
            "name": "CNN",
            "structure": structure_def,
            "optimizer": tf.train.AdamOptimizer(learning_rate=0.001),
            "cost_function": tf.nn.softmax_cross_entropy_with_logits,
            "num_epochs": 100,
            "batch_size": 100,
            "model_dir": model_dir,
            "epoch_log_verbosity": 1,
            "early_stopping_metric": "val_accuracy",
            "early_stopping_min_delta": 0.01,
            "early_stopping_patience": 3
        }

        model_configs.append(model_config)

        # Save model config
        with open(model_dir + "/cnn_config.pkl", "wb") as f:
            pickle.dump(model_config, f, protocol=2)
            print("Saved config to model directory")

    # Setup GridSearch --> specify name of class and the model configs
    searcher = GridSearch(CNN, model_configs)
    best_score, best_config = searcher.run(train_dataset,
                                           valid_dataset)

    """
    def __init__(self, model_class, model_configs, metric="accuracy",
                 maximize=True):
        """
        Initialize grid search

        Args:
        model_class: Reference to the class name of the models
        model_configs (list): List of model config dictionaries
        metric (str): Which metric to select for
        maximize (bool): Find a the config that either maximizes or minimizes
                         the metric
        """
        self.model_class = model_class
        self.model_configs = model_configs
        self.metric = metric
        self.maximize = True

        self.best_score = 0.0
        self.best_config = {}

        self.logger = logging.getLogger(self.__class__.__name__)

        if len(self.logger.handlers) == 0:
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(name)s] %(asctime)s - %(message)s")

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.info("Initalized grid search with {0} cases"
                         .format(len(model_configs)))

    def run(self, train_dataset, validation_dataset):
        for index, config in enumerate(self.model_configs):
            self.logger.info("Case #{0}:".format(index + 1))
            self.logger.info(str(config) + "\n")

            config["name"] = config["name"] + "_{0}".format(index + 1)

            with tf.Session() as sess:
                model = self.model_class(sess, config)
                model.fit(train_dataset, validation_dataset)
                valid_cost, valid_results = model.evaluate(validation_dataset)

            # Clear default graph (remove tensors and ops)
            tf.reset_default_graph()

            score = getattr(valid_results, self.metric)()
            self.logger.info("Score: {0} \n".format(score))
            if self.maximize:
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = config
            else:
                if score < self.best_score:
                    self.best_score = score
                    self.best_config = config

        return (self.best_score, self.best_config)
