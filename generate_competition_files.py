# imports
from utils.DependencyParser import DependencyParser

if __name__ == '__main__':
    path_to_train_file = './data/train.labeled'
    path_to_test_file = './data/test.labeled'
    path_to_comp_file = './data/comp.unlabeled'
    parser_1 = DependencyParser(path_to_train_file, minimal=True, path_to_valid_file=path_to_test_file,
                                use_mcdonald=False, feature_threshold=0, path_to_weights='./data/Minimal/')  # TODO

    parser_2 = DependencyParser(path_to_train_file, minimal=False, path_to_valid_file=path_to_test_file,
                                use_mcdonald=True, feature_threshold=1, path_to_weights='./data/Full/')  # TODO
    print("parsing using model 1")
    parser_1.generate_labeled_file(path_to_comp_file, name='_m1_id')

    print("parsing using model 2")
    parser_2.generate_labeled_file(path_to_comp_file, name='_m2_id')
