# imports
import time
from utils.chu_liu import Digraph
import numpy as np
from copy import deepcopy
from typing import List
import os
from utils.utils import dep_sample_generator,\
    sample_to_full_successors, successors_to_sample,\
    DepSample, sample_to_successors
from utils.features import generate_unigram_feat_dict, generate_bigram_feat_dict_minimal, generate_bigram_feat_dict, \
    extract_unigram_bigram_feat_indices_pair, extract_unigram_bigram_feat_indices
from utils.DepOptimizer import DepOptimizer


class DependencyParser:
    """
    This class represents our dependency trees parsing model.
    It will be initialized with a given training file and give the following API:
    .infer(sentence: str)
    .perceptron_train(num_iterations)
    .load_weights(path_to_weights: str)
    ...
    """

    def __init__(self, path_to_train_file: str, minimal: bool = True, path_to_valid_file=None):
        """
        :param path_to_train_file: training file that contains the samples (str)
        :param path_to_valid_file: validation file that contains the samples (str)
        :param minimal: whether or not to use the minimal version of the features (bool)
        """

        self.training_file_path = path_to_train_file
        self.minimal = minimal
        self.path_to_valid_file = path_to_valid_file
        if minimal:
            self.dicts = [generate_unigram_feat_dict(path_to_train_file),
                          generate_bigram_feat_dict_minimal(path_to_train_file)]
        else:
            self.dicts = [generate_unigram_feat_dict(path_to_train_file),
                          generate_bigram_feat_dict(path_to_train_file)]

        self.feature_extractor = extract_unigram_bigram_feat_indices

        self.num_of_features = np.sum([len(k) for d in self.dicts for k in d])
        print("total number features in the model: ", self.num_of_features)

        self.weights_file_name = os.path.join(os.path.dirname(self.training_file_path), '.weights')

        if os.path.isfile(self.weights_file_name):
            self.w = np.load(self.weights_file_name)
            print("loaded weights from ", self.weights_file_name)
        else:
            self.w = np.zeros(self.num_of_features)  # try to load or init to 0
            print("initialized weights to zeros")
        self.dep_weights = DepOptimizer(self.w, None, path_to_train_file=self.training_file_path,
                                        dicts=self.dicts, minimal=self.minimal)

    def perceptron_train(self, num_iterations: int, accuracy_step=10)-> None:
        """
        Given the number of iterations for training we loop
        over the training file said number of iterations preforming
        the perceptron algorithm
        the result is updated weights in self.w
        :param num_iterations: number of iterations to perform (int)
        :param accuracy_step: interval between accuracy calculation (int)
        :return: None
        """
        print("training started")
        self.w = np.zeros(self.num_of_features)
        st_time = time.time()
        # dep_weights = DepOptimizer(self.w, None, path_to_train_file=self.training_file_path,
        #                            dicts=self.dicts, minimal=self.minimal) # moved to class level
        for i in range(num_iterations):
            total_sentences = 0
            correct_sentences = 0
            total_words = 0
            correct_words = 0
            it_st_time = time.time()
            for sample in dep_sample_generator(self.training_file_path):
                total_sentences += 1
                sample_len = sample[-1].idx
                successors = sample_to_full_successors(sample_len)
                # TODO: consider moving DepOptimizer out of the loop and just use update_weights, update_sample
                # dep_weights = DepOptimizer(self.w, sample, dicts=self.dicts, minimal=self.minimal)
                self.dep_weights.update_sample(sample)
                self.dep_weights.update_weights(self.w)
                graph = Digraph(successors, self.dep_weights.get_score)
                argmax_tree = graph.mst().successors
                argmax_tree = {k: v for k, v in argmax_tree.items() if v}
                infered_sample = successors_to_sample(deepcopy(sample), argmax_tree)
                for j in range(len(sample)):
                    if not j:
                        # skip ROOT
                        continue
                    total_words += 1
                    if sample[j].head == infered_sample[j].head:
                        correct_words += 1
                ground_truth_successors = sample_to_successors(sample)

                #  according to python doc dictionary == works as expected
                #  returning true only if both have same keys and same values to those keys
                #  order of dict.values() corresponded to dict.keys()
                if argmax_tree != ground_truth_successors:
                    features_ground_truth = self.feature_extractor(sample, self.dicts, self.minimal)
                    features_argmax = self.feature_extractor(infered_sample,
                                                             self.dicts, self.minimal)
                    self.w[list(features_ground_truth.keys())] += np.array(list(features_ground_truth.values()))
                    self.w[list(features_argmax.keys())] -= np.array(list(features_argmax.values()))
                else:
                    correct_sentences += 1
            print('iteration/epoch ', i, "- iteration time: %.2f min" % ((time.time() - it_st_time) / 60),
                  ", train accuracy:: sentence: %.3f " % (1.0 * correct_sentences / total_sentences),
                  " words: %.3f " % (1.0 * correct_words / total_words),
                  ", total time: %.2f min" % ((time.time() - st_time) / 60))
            if i % accuracy_step == 0 and self.path_to_valid_file is not None:
                print("validation accuracy calculation step:")
                valid_sent_acc, valid_word_acc = self.calc_accuracy(self.path_to_valid_file)
                print("valid accuracy:: sentence: %.3f" % valid_sent_acc, " words: %.3f" % valid_word_acc)
                self.w.dump(self.weights_file_name)
                print("saved weights @ ", self.weights_file_name)

        self.w.dump(self.weights_file_name)

    def infer(self, sentence: List[DepSample])-> List[DepSample]:
        """
        given a sentence we run chu_lie graph inference
        and return a list of DepSample with parsed sentence heads
        :param sentence: list of DepSample ( we deep copy it)
        :return: list of DepSample with inferred heads
        """
        sample_len = sentence[-1].idx
        successors = sample_to_full_successors(sample_len)
        self.dep_weights.update_sample(sentence)
        # dep_weights = DepOptimizer(self.w, sentence)

        graph = Digraph(deepcopy(successors), self.dep_weights.get_score)
        tree = graph.mst().successors
        return successors_to_sample(deepcopy(sentence), tree)

    def calc_accuracy(self, path_test_file):
        """
        This function calculates both sentence accuracy and word accuracy on a given test set.
        :param path_test_file: path to file containing labeled samples (str)
        :return: sentence_accuracy: percentage on complete sentences (float)
        :return: word_accuracy: percentage on words (float)
        """
        total_words = 0
        total_sentences = 0
        correct_words = 0
        correct_sentences = 0
        samp_gen = dep_sample_generator(path_test_file)
        for sample in samp_gen:
            total_sentences += 1
            total_words += sample[-1].idx
            infered_sample = self.infer(sample)
            correct_parse = True
            for i in range(len(sample)):
                if not i:
                    # skip ROOT
                    continue
                if sample[i].head == infered_sample[i].head:
                    correct_words += 1
                else:
                    correct_parse = False
            if correct_parse:
                correct_sentences += 1
        sentence_accuracy = 1.0 * correct_sentences / total_sentences
        word_accuracy = 1.0 * correct_words / total_words
        return sentence_accuracy, word_accuracy