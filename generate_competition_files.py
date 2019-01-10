# imports
from utils.DependencyParser import DependencyParser

if __name__ == '__main__':
    path_to_train_file = './data/train.labeled'
    path_to_test_file = './data/test.labeled'
    path_to_comp_file = './data/comp.unlabeled'
    parser_1 = DependencyParser(path_to_train_file, minimal=True, path_to_valid_file=path_to_test_file,
                                use_mcdonald=False, feature_threshold=0,
                                path_to_weights='./pretrained/model1best.weights')

    parser_2 = DependencyParser(path_to_train_file, minimal=False, path_to_valid_file=path_to_test_file,
                                use_mcdonald=True, feature_threshold=1,
                                path_to_weights='./pretrained/model2best.weights')

    print("validating model 1 integrity")
    s_a, w_a = parser_1.calc_accuracy(path_to_test_file)
    print("integrity:: sentence: %.3f" % s_a, " words: %.3f" % w_a)
    print("parsing using model 1")
    m_1_labeled_path = parser_1.generate_labeled_file(path_to_comp_file, name='_m1_id')
    print("validating model 1 inference file")
    s_a, w_a = parser_1.calc_accuracy(m_1_labeled_path)
    print("validation:: sentence: %.3f" % s_a, " words: %.3f" % w_a)

    print("validating model 2 integrity")
    s_a, w_a = parser_2.calc_accuracy(path_to_test_file)
    print("integrity:: sentence: %.3f" % s_a, " words: %.3f" % w_a)
    print("parsing using model 2")
    m_2_labeled_path = parser_2.generate_labeled_file(path_to_comp_file, name='_m2_id')
    print("validating model 2 inference file")
    s_a, w_a = parser_2.calc_accuracy(m_2_labeled_path)
    print("integrity:: sentence: %.3f" % s_a, " words: %.3f" % w_a)
