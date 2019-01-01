# imports
import time

import numpy as np
from utils.features import generate_unigram_feat_dict, generate_bigram_feat_dict_minimal, generate_bigram_feat_dict, \
    extract_unigram_bigram_feat_indices_pair, extract_unigram_bigram_feat_indices


class DepOptimizer:
    """
    This helper class holds the weights of a parser model and given a sentence,
    calculates scores for edges.
    """

    def __init__(self, w, sample, path_to_train_file=None, dicts=None, feature_extractor=None, minimal=True):
        """
        Initialize an optimizer.
        :param: w: weights (list)
        :param: sample: current sample (list of DepSample)
        :param: path_to_train_file: training file that contains the samples (str)
        :param: dicts: dictionaries of features (list of dicts)
        :param: feature_extractor: function to extract feature for (head, child)
        :param: minimal: whether or not to use the minimal version of the features
        """
        self.w = w
        self.sample = sample
        self.minimal = minimal
        if path_to_train_file is None:
            self.path_to_train_file = './data/train.labeled'
        else:
            self.path_to_train_file = path_to_train_file

        self.dicts = dicts
        if dicts is None:
            if minimal:
                self.dicts = [generate_unigram_feat_dict(self.path_to_train_file),
                              generate_bigram_feat_dict_minimal(self.path_to_train_file)]
            else:
                self.dicts = [generate_unigram_feat_dict(self.path_to_train_file),
                              generate_bigram_feat_dict(self.path_to_train_file)]
        if feature_extractor is None:
            self.feature_extractor = extract_unigram_bigram_feat_indices_pair
        else:
            self.feature_extractor = feature_extractor

    def get_score(self, head_int, child_int):
        """
        Calculates a score for an edge between `head_int` to `child_int`.
        :param: head_int: head node id (int)
        :param: child_int: child node id (int)
        :return: score: score for the edge (float)
        """
        features_inds = self.feature_extractor(self.sample[head_int], self.sample[child_int], self.dicts, self.minimal)
        w = np.array(self.w)
        return np.sum(w[features_inds])

    def update_weights(self, w):
        """
        Updates the optimizer current weights.
        :param: w: weughts (list)
        """
        self.w = w

    def update_sample(self, sample):
        """
        Updates current sample.
        """
        self.sample = sample

