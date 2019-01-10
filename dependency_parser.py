# imports
from utils.DependencyParser import DependencyParser
import numpy as np

if __name__ == '__main__':
    path_to_train_file = './data/train_m2.labeled'
    # path_to_valid_file = './data/train.labeled.valid.labeled'
    path_to_test_file = './data/test_m2.labeled'
    parser = DependencyParser(path_to_train_file, minimal=False, path_to_valid_file=path_to_test_file,
                              use_mcdonald=True, feature_threshold=1)
    parser.perceptron_train(100, 2)

    path_to_train_file = './data/train_m1.labeled'
    # path_to_valid_file = './data/train.labeled.valid.labeled'
    path_to_test_file = './data/test_m1.labeled'
    parser = DependencyParser(path_to_train_file, minimal=True, path_to_valid_file=path_to_test_file,
                              use_mcdonald=False, feature_threshold=0)
    parser.perceptron_train(100, 2)
