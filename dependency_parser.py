# imports
from utils.DependencyParser import DependencyParser


if __name__ == '__main__':
    path_to_train_file = './data/train.labeled.train.labeled'
    path_to_valid_file = './data/train.labeled.valid.labeled'
    parser = DependencyParser(path_to_train_file, minimal=False, path_to_valid_file=path_to_valid_file)
    parser.perceptron_train(100)
