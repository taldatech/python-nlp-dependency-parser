# imports
import time
from utils.chu_liu import Digraph
import numpy as np
from copy import deepcopy
from typing import List
import os
from utils.ProgressBar import  ProgressBar
from utils.features_v2 import extract_global_features, generate_features_dicts, extract_local_feature_indices, \
    generate_global_features_dict
from utils.utils import dep_sample_generator, \
    sample_to_full_successors, successors_to_sample, \
    DepSample, sample_to_successors, generate_fully_connected_graphs, generate_ground_truth_trees, \
    sample_to_lines
from utils.features import generate_unigram_feat_dict, generate_bigram_feat_dict_minimal, generate_bigram_feat_dict, \
    extract_unigram_bigram_feat_indices_pair, extract_unigram_bigram_feat_indices, ROOT
from utils.DepOptimizer import DepOptimizer
from collections import namedtuple
import pickle


class DependencyParser:
    """
    This class represents our dependency trees parsing model.
    It will be initialized with a given training file and give the following API:
    .infer(sentence: str)
    .perceptron_train(num_iterations)
    .load_weights(path_to_weights: str)
    ...
    """

    def __init__(self, path_to_train_file: str, minimal: bool = True, path_to_valid_file=None, use_mcdonald=True):
        """
        :param path_to_train_file: training file that contains the samples (str)
        :param path_to_valid_file: validation file that contains the samples (str)
        :param minimal: whether or not to use the minimal version of the features (bool)
        :param use_mcdonald: whether or not to use features from McDonald's paper
        """

        self.training_file_path = path_to_train_file
        self.minimal = minimal
        self.use_mcdonald = use_mcdonald
        self.path_to_valid_file = path_to_valid_file
        # if minimal:
        #     self.dicts = [generate_unigram_feat_dict(path_to_train_file),
        #                   generate_bigram_feat_dict_minimal(path_to_train_file)]
        # else:
        #     self.dicts = [generate_unigram_feat_dict(path_to_train_file),
        #                   generate_bigram_feat_dict(path_to_train_file)]

        self.dicts = generate_features_dicts(path_to_train_file, minimal=minimal, use_mcdonald=use_mcdonald)

        self.feature_extractor = extract_global_features
        self.fc_graphs = generate_fully_connected_graphs(path_to_train_file)
        self.gt_trees = generate_ground_truth_trees(path_to_train_file)
        self.gt_global_features = generate_global_features_dict(path_to_train_file, self.feature_extractor, self.dicts,
                                                                save_to_file=True, minimal=self.minimal,
                                                                use_mcdonald=use_mcdonald)

        # self.num_of_features = np.sum([len(k) for d in self.dicts for k in d])
        self.num_of_features = np.sum([len(d) for d in self.dicts.values()])
        print("total number features in the model: ", self.num_of_features)

        self.weights_file_name = self.training_file_path + '.weights'

        if os.path.isfile(self.weights_file_name):
            self.w = np.load(self.weights_file_name)
            print("loaded weights from ", self.weights_file_name)
        else:
            self.w = np.zeros(self.num_of_features)  # try to load or init to 0
            print("initialized weights to zeros")
        self.dep_weights = DepOptimizer(self.w, None, path_to_train_file=self.training_file_path,
                                        dicts=self.dicts, minimal=self.minimal,
                                        feature_extractor=extract_local_feature_indices,
                                        use_mcdonald=use_mcdonald)

    def perceptron_train(self, num_iterations: int, accuracy_step=10) -> None:
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
        num_samples = 0
        for _ in dep_sample_generator(self.training_file_path):
            num_samples += 1
        st_time = time.time()
        # dep_weights = DepOptimizer(self.w, None, path_to_train_file=self.training_file_path,
        #                            dicts=self.dicts, minimal=self.minimal) # moved to class level
        train_word_accuracies = []
        train_sentenence_accuracies = []
        for i in range(num_iterations):
            print("iteration: ", i)
            progress = ProgressBar(num_samples, fmt=ProgressBar.FULL)
            total_sentences = 0
            correct_sentences = 0
            total_words = 0
            correct_words = 0
            it_st_time = time.time()
            for idx, sample in enumerate(dep_sample_generator(self.training_file_path)):
                total_sentences += 1
                sample_len = sample[-1].idx

                successors = self.fc_graphs[sample_len]  # sample_to_full_successors(sample_len)
                # dep_weights = DepOptimizer(self.w, sample, dicts=self.dicts, minimal=self.minimal)
                self.dep_weights.update_sample(sample)
                self.dep_weights.update_weights(self.w)
                graph = Digraph(successors, self.dep_weights.get_score)
                mst_start_time = time.time()
                argmax_tree = graph.mst().successors
                argmax_tree = {k: v for k, v in argmax_tree.items() if v}
                ground_truth_successors = self.gt_trees[idx]  # sample_to_successors(sample)

                # print("mst calc time: %.5f secs" % (time.time() - mst_start_time))
                infered_sample = successors_to_sample(deepcopy(sample), argmax_tree)
                for j in range(len(sample)):
                    if not j:
                        # skip ROOT
                        continue
                    total_words += 1
                    if sample[j].head == infered_sample[j].head:
                        correct_words += 1

                #  according to python doc dictionary == works as expected
                #  returning true only if both have same keys and same values to those keys
                #  order of dict.values() corresponded to dict.keys()
                if argmax_tree != ground_truth_successors:
                    # features_ground_truth = self.feature_extractor(sample, self.dicts, self.minimal)
                    #  could also be replaced by a dict
                    features_ground_truth = self.gt_global_features[idx]
                    feat_calc_start_time = time.time()
                    features_argmax = self.feature_extractor(infered_sample,
                                                             self.dicts, self.minimal, use_mcdonald=self.use_mcdonald)
                    # print("feature extraction time: %.5f" % (time.time() - feat_calc_start_time))
                    self.w[list(features_ground_truth.keys())] += np.array(list(features_ground_truth.values()))
                    self.w[list(features_argmax.keys())] -= np.array(list(features_argmax.values()))

                else:
                    correct_sentences += 1
                progress.current += 1
                progress()
            sen_acc = 1.0 * correct_sentences / total_sentences
            word_acc = 1.0 * correct_words / total_words
            train_sentenence_accuracies.append(sen_acc)
            train_word_accuracies.append(word_acc)
            progress.done()
            print('iteration/epoch ', i, "- iteration time: %.2f min" % ((time.time() - it_st_time) / 60),
                  ", train accuracy:: sentence: %.3f " % sen_acc,
                  " words: %.3f " % word_acc,
                  ", total time: %.2f min" % ((time.time() - st_time) / 60))

            if (i + 1) % accuracy_step == 0 and self.path_to_valid_file is not None:
                print("validation accuracy calculation step:")
                valid_sent_acc, valid_word_acc = self.calc_accuracy(self.path_to_valid_file)
                print("valid accuracy:: sentence: %.3f" % valid_sent_acc, " words: %.3f" % valid_word_acc)
                self.w.dump(self.weights_file_name)
                print("saved weights @ ", self.weights_file_name)
                # save checkpoint
                path = self.training_file_path + "_epoch_" + str(i) + ".checkpoint"
                ckpt = {}
                ckpt['weights'] = self.w.tolist()
                ckpt['train_acc'] = (sen_acc, word_acc)
                ckpt['valid_acc'] = (valid_sent_acc, valid_word_acc)
                with open(path, 'wb') as fp:
                    pickle.dump(ckpt, fp)
                print("saved checkpoint @ ", path)

        self.w.dump(self.weights_file_name)
        path = self.training_file_path + "_" + str(i + 1) + "_epochs" + ".results"
        ckpt = {}
        ckpt['weights'] = self.w.tolist()
        ckpt['train_word_acc'] = train_word_accuracies
        ckpt['train_sen_acc'] = train_sentenence_accuracies
        with open(path, 'wb') as fp:
            pickle.dump(ckpt, fp)
        print("saved final results @ ", path)

    def infer(self, sentence: List[DepSample]) -> List[DepSample]:
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

    def generate_labeled_file(self, path_to_unlabeled_file):
        """
        This function generates labels for unlabeled samples in the same
        format as the original file.
        :param: path_to_unlabeled_file: path to location of the file (str)
        """
        root = DepSample(0, ROOT, ROOT, 0)
        path_to_labeled = path_to_unlabeled_file + '.labeled'
        with open(path_to_labeled, 'w') as fw:
            with open(path_to_unlabeled_file) as fr:
                sample = [root]
                lines = []
                for line in fr:
                    if not line.rstrip():
                        # end of sample
                        infered_sample = self.infer(sample)
                        # infered_sample = sample
                        res_lines = sample_to_lines(infered_sample, lines)
                        for l in res_lines:
                            fw.write(l)
                            fw.write('\n')
                        fw.write('\n')
                        sample = [root]
                        lines = []
                    else:
                        lines.append(line)
                        ls = line.rstrip().split('\t')
                        try:
                            head = int(ls[6])
                        except ValueError:
                            head = ls[6]
                        sample.append(DepSample(int(ls[0]), ls[1], ls[3], head))
        print("finished generating labeled file of ", path_to_unlabeled_file, " @ ", path_to_labeled)
