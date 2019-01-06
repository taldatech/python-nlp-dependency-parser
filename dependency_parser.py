# imports
from utils.DependencyParser import DependencyParser
import numpy as np

if __name__ == '__main__':
    path_to_train_file = './data/train.labeled.train.labeled'
    path_to_valid_file = './data/train.labeled.valid.labeled'
    path_to_test_file = './data/test.labeled'
    parser = DependencyParser(path_to_train_file, minimal=True, path_to_valid_file=path_to_valid_file,
                              use_mcdonald=True, feature_threshold=1)
    parser.perceptron_train(100, 2)
    # parser.w = np.load(r'./data/tal.weights')
    # parser.dep_weights.update_weights(parser.w)
    # print('sentence_acc{:} word_acc{:}'.format(*parser.calc_accuracy(path_to_test_file)))
