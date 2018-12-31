import numpy as np
import pandas as pd
import os
from typing import List
from copy import deepcopy
from utils.chu_liu import Digraph
from utils.utils import dep_sample_generator,\
    sample_to_full_successors, successors_to_sample,\
    DepSample, sample_to_successors


class DependencyParser:
    """
    this class is to represent our dependency trees parsing model
    it will be initialized given a training file and give the following api
    .infer(sentence: str)
    .preceptron_train(num_iterations)
    .load_weights(path_to_weights: str)
    ...
    """

    def __init__(self, path_to_train_file: str):
        """

        :param path_to_train_file:
        """

        self.training_file_path = path_to_train_file
        # self.dicts = generate feature dicts
        self.num_of_features = 65000  #_calc_num_of_features()

        self.weights_file_name = os.path.join(os.path.dirname(self.training_file_path), '_weights.ndarray')

        if os.path.isfile(self.weights_file_name):
            self.w = np.load(self.weights_file_name)
        else:
            self.w = np.zeros(self.num_of_features)  # try to load or init to 0

    def preceptron_train(self, num_iterations: int)-> None:
        """
        given the number of iterations for training we loop
        over the training file said number of iterations preforming
        the preceptron algorithm
        the result is updated weights in self.w
        :return: None
        """
        self.w = np.zeros(self.num_of_features)
        for i in range(num_iterations):

            for sample in dep_sample_generator(self.training_file_path):

                sample_len = sample[-1].idx
                successors = sample_to_full_successors(sample_len)
                dep_weights = DepOptimizer(self.w, sample)

                graph = Digraph(successors, dep_weights.get_score)
                argmax_tree = graph.mst().successors

                ground_truth_successors = sample_to_successors(sample)

                #  according to python doc dictionary == works as expected
                #  returning true only if both have same keys and same values to those keys
                #  order of dict.values() corresponded to dict.keys()
                if argmax_tree != ground_truth_successors:
                    features_ground_truth = {0: 1}  # TODO extract_unigram_bigram_feat_indices(sample, self.dicts)
                    features_argmax = {3: 2}  # TODO extract_unigram_bigram_feat_indices(successors_to_sample(sample, argmax_tree), self.dicts)
                    self.w[list(features_ground_truth.keys())] += np.array(list(features_ground_truth.values()))
                    self.w[list(features_argmax.keys())] += np.array(list(features_argmax.values()))

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
        dep_weights = DepOptimizer(self.w, sentence)

        graph = Digraph(deepcopy(successors), dep_weights.get_score)
        tree = graph.mst().successors
        return successors_to_sample(deepcopy(sentence), tree)


class DepOptimizer:
    """
    This helper class holds the weights of a parser model and given a sentence,
    calculates scores for edges.
    """

    def __init__(self, w, sample):
        """
        Initialzie an optimzier.
        :param: w: weights (list)
        :param: sample: current sample (list of DepSample)
        """
        self.w = w
        self.sample = sample
        # add dicts ?

    def get_score(self, head_int, child_int):
        """
        Calculates a score for an edge between `head_int` to `child_int`.
        :param: head_int: head node id (int)
        :param: child_int: child node id (int)
        :return: score: score for the edge (float)
        """
        # TODO: implemenet np.sum(w[extract_local_features(self.sample[head_int], self.sample[child_int], self.dicts)])
        return 0

    def update_weights(self, w):
        """
        Updates the optimizer current weights.
        :param: w: weughts (list)
        """
        self.w = w

    def update_sample(self, sample):
        """
        Upates current sample.
        """
        self.sample = sample


if __name__ == '__main__':

    path_to_file = './data/train.labeled'
    parser = DependencyParser(path_to_file)
    parser.preceptron_train(5)
