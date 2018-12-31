import time
import os
import pickle
import numpy as np
from collections import OrderedDict
from collections import namedtuple
from random import shuffle
import copy
from itertools import combinations, combinations_with_replacement
# named tuple has methods like _asdict()

"""
UNIGRAMS + BIGRAMS
"""


def extract_unigram_bigram_feat_indices(sample, dicts, minimal=False):
    """
    This function extracts the indices (in the feature vector) of the features:
    * (head_word, head_pos)
    * (head_word)
    * (head_pos)
    * (child_word, child_pos)
    * (child_word)
    * (child_pos)
    * (head_word, head_pos, child_word, child_pos)
    * (head_pos, child_word, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_word)
    * (head_word, child_word)
    * (head_pos, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: dicts: the dictionaries of indices [unigram_dicts, bigrams_dicts] (list)
    :param: minimal: whether or not to use the minimal version (bool)
    :return: feat_indices_dict: dictionary idx->count
    """
    unigram_dict, bigram_dict = dicts[0], dicts[1]
    unigram_inds = extract_unigram_feat_indices(sample, unigram_dict)
    feat_indices_dict = copy.deepcopy(unigram_inds)
    if minimal:
        bigram_inds = extract_bigram_feat_indices_minimal(sample, bigram_dict)
    else:
        bigram_inds = extract_bigram_feat_indices(sample, bigram_dict)
    update_dict(feat_indices_dict, bigram_inds)
    return OrderedDict(sorted(feat_indices_dict.items(), key=lambda t: t[0]))


def generate_word_hist_dict(path_to_file, save_to_file=False):
    """
    This function generates histogram of of the tokens in the dataset.
    :param: path_to_file: path to location of the dataset (str)
    :param: save_to_file: whether or not to save the dictionary on disk (bool)
    :return: word_hist: OrderedDict word->word_count
    """
    samp_gen = dep_sample_generator(path_to_file)
    word_hist = {}
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT:
                continue
            if word_hist.get(s.token):
                word_hist[s.token] += 1
            else:
                word_hist[s.token] = 1
    word_hist = OrderedDict(sorted(word_hist.items(), key=lambda t: -t[1]))
    if save_to_file:
        path = path_to_file + ".word.hist"
        with open(path, 'wb') as fp:
            pickle.dump(word_hist, fp)
        print("word histogram dictionary saved @ ", path)
    return word_hist

"""
UNIGRAMS
"""

def generate_hw_hp_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, head_pos)
    * (head_word)
    * (head_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hw_hp_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hw_hp_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            # ignore ROOT
            if s.token == ROOT or (word_hist.get(sample[s.head].token) \
                                   and word_hist[sample[s.head].token] < word_threshold):
                continue
            if sample[s.head].token == ROOT:
                continue
            feats = [(sample[s.head].token, sample[s.head].pos),
                     (sample[s.head].token),
                     (sample[s.head].pos)]
            for feat in feats:
                if hw_hp_feat_dict.get(feat) is None:
                    hw_hp_feat_dict[feat] = current_idx
                    current_idx += 1
    print("total (head_word, head_pos), (head_word), (head_pos) features: ", current_idx)
    hw_hp_feat_dict = OrderedDict(sorted(hw_hp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hw_hp.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hw_hp_feat_dict, fp)
        print("saved (head_word, head_pos), (head_word), (head_pos) features dictionary @ ", path)
    return hw_hp_feat_dict


def extract_hw_hp_feat_indices(sample, hw_hp_dict):
    """
    This function extracts the indices (in the feature vector) of the unigrams features:
    * (head_word, head_pos)
    * (head_word)
    * (head_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hw_hp_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hw_hp_dict.get((sample[s.head].token, sample[s.head].pos)):
            idx = hw_hp_dict.get((sample[s.head].token, sample[s.head].pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
        if hw_hp_dict.get((sample[s.head].token)):
            idx = hw_hp_dict.get((sample[s.head].token))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
        if hw_hp_dict.get((sample[s.head].pos)):
            idx = hw_hp_dict.get((sample[s.head].pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hw_hp_feat_indices_pair(head, child, hw_hp_dict):
    """
    This function extracts the indices (in the feature vector) of the unigrams features:
    * (head_word, head_pos)
    * (head_word)
    * (head_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hw_hp_dict: the dictionary of indices (dict)
    :return: feat_indices_list: list of indices
    """
    feat_indices = []
    if hw_hp_dict.get((head.token, head.pos)):
        feat_indices.append(hw_hp_dict.get((head.token, head.pos)))
    if hw_hp_dict.get(head.token):
        feat_indices.append(hw_hp_dict.get(head.token))
    if hw_hp_dict.get(head.pos):
        feat_indices.append(hw_hp_dict.get(head.pos))
    return feat_indices


def generate_cw_cp_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (child_word, child_pos)
    * (child_word)
    * (child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: cw_cp_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    cw_cp_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feats = [(s.token, s.pos),
                     (s.token),
                     (s.pos)]
            for feat in feats:
                if cw_cp_feat_dict.get(feat) is None:
                    cw_cp_feat_dict[feat] = current_idx
                    current_idx += 1
    print("total (child_word, child_pos), (child_word), (child_pos) features: ", current_idx)
    cw_cp_feat_dict = OrderedDict(sorted(cw_cp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".cw_cp.dict"
        with open(path, 'wb') as fp:
            pickle.dump(cw_cp_feat_dict, fp)
        print("saved (child_word, child_pos), (child_word), (child_pos) features dictionary @ ", path)
    return cw_cp_feat_dict


def extract_cw_cp_feat_indices(sample, cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the unigrams features:
    * (child_word, child_pos)
    * (child_word)
    * (child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: cw_cp_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if cw_cp_dict.get((s.token, s.pos)):
            idx = cw_cp_dict.get((s.token, s.pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
        if cw_cp_dict.get((s.token)):
            idx = cw_cp_dict.get((s.token))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
        if cw_cp_dict.get((s.pos)):
            idx = cw_cp_dict.get((s.pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_cw_cp_feat_indices_pair(head, child, cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the unigrams features:
    * (child_word, child_pos)
    * (child_word)
    * (child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: cw_cp_dict: the dictionary of indices (dict)
    :return: feat_indices_list: list of indices
    """
    feat_indices = []
    if cw_cp_dict.get((child.token, child.pos)):
        feat_indices.append(cw_cp_dict.get((child.token, child.pos)))
    if cw_cp_dict.get(child.token):
        feat_indices.append(cw_cp_dict.get(child.token))
    if cw_cp_dict.get(child.pos):
        feat_indices.append(cw_cp_dict.get(child.pos))
    return feat_indices


def generate_unigram_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, head_pos)
    * (head_word)
    * (head_pos)
    * (child_word, child_pos)
    * (child_word)
    * (child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: unigram_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    hw_hp_dict = generate_hw_hp_feat_dict(path_to_file, word_threshold=word_threshold,
                                          save_to_file=save_to_file, word_hist=word_hist)
    cw_cp_dict = generate_cw_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                          save_to_file=save_to_file, word_hist=word_hist)
    print("total unigrams features: ", len(hw_hp_dict) + len(cw_cp_dict))
    return hw_hp_dict, cw_cp_dict


def extract_unigram_feat_indices_pair(head, child, unigram_dict):
    """
    This function extracts the indices (in the feature vector) of the unigrams features:
    * (head_word, head_pos)
    * (head_word)
    * (head_pos)
    * (child_word, child_pos)
    * (child_word)
    * (child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: unigram_dict: the dictionaries of indices (dict)
    :return: feat_indices_list: list of indices
    """
    num_hw_hp_feats = len(unigram_dict[0])
    num_cw_cp_feats = len(unigram_dict[1])
    current_num_features = 0
    hw_hp_ind = extract_hw_hp_feat_indices_pair(head, child, unigram_dict[0])
    current_num_features += num_hw_hp_feats
    cw_cp_ind = extract_cw_cp_feat_indices_pair(head, child, unigram_dict[1])
    unigram_indices = copy.deepcopy(hw_hp_ind)
    for i in cw_cp_ind:
        unigram_indices.append(current_num_features + i)
    current_num_features += num_cw_cp_feats
    return sorted(unigram_indices)


def extract_unigram_feat_indices(sample, unigram_dict):
    """
    This function extracts the indices (in the feature vector) of the unigrams features:
    * (head_word, head_pos)
    * (head_word)
    * (head_pos)
    * (child_word, child_pos)
    * (child_word)
    * (child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: unigram_dict: the dictionaries of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    num_hw_hp_feats = len(unigram_dict[0])
    num_cw_cp_feats = len(unigram_dict[1])
    current_num_features = 0
    hw_hp_ind = extract_hw_hp_feat_indices(sample, unigram_dict[0])
    current_num_features += num_hw_hp_feats
    cw_cp_ind = extract_cw_cp_feat_indices(sample, unigram_dict[1])
    unigram_indices = copy.deepcopy(hw_hp_ind)
    for item in cw_cp_ind.items():
        unigram_indices[current_num_features + item[0]] = item[1]
    current_num_features += num_cw_cp_feats
    return OrderedDict(sorted(unigram_indices.items(), key=lambda t: t[0]))


"""
BIGRAMS
"""


def generate_hw_hp_cw_cp_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, head_pos, child_word, child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
            that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hw_hp_cw_cp_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hw_hp_cw_cp_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feat = (sample[s.head].token, sample[s.head].pos, s.token, s.pos)
            if hw_hp_cw_cp_feat_dict.get(feat) is None:
                hw_hp_cw_cp_feat_dict[feat] = current_idx
                current_idx += 1
    print("total (head_word, head_pos, child_word, child_pos) features: ", current_idx)
    hw_hp_cw_cp_feat_dict = OrderedDict(sorted(hw_hp_cw_cp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hw_hp_cw_cp.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hw_hp_cw_cp_feat_dict, fp)
        print("saved (head_word, head_pos, child_word, child_pos) features dictionary @ ", path)
    return hw_hp_cw_cp_feat_dict


def extract_hw_hp_cw_cp_feat_indices(sample, hw_hp_cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_word, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hw_hp_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hw_hp_cw_cp_dict.get((sample[s.head].token, sample[s.head].pos, s.token, s.pos)):
            idx = hw_hp_cw_cp_dict.get((sample[s.head].token, sample[s.head].pos, s.token, s.pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hw_hp_cw_cp_feat_indices_pair(head, child, hw_hp_cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_word, child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hw_hp_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_idx: index of the feature (list)
    """
    if hw_hp_cw_cp_dict.get((head.token, head.pos, child.token, child.pos)):
        return [hw_hp_cw_cp_dict.get((head.token, head.pos, child.token, child.pos))]
    else:
        return []


def generate_hp_cw_cp_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_pos, child_word, child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hp_cw_cp_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hp_cw_cp_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feat = (sample[s.head].pos, s.token, s.pos)
            if hp_cw_cp_feat_dict.get(feat) is None:
                hp_cw_cp_feat_dict[feat] = current_idx
                current_idx += 1
    print("total (head_pos, child_word, child_pos) features: ", current_idx)
    hp_cw_cp_feat_dict = OrderedDict(sorted(hp_cw_cp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hp_cw_cp.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hp_cw_cp_feat_dict, fp)
        print("saved (head_pos, child_word, child_pos) features dictionary @ ", path)
    return hp_cw_cp_feat_dict


def extract_hp_cw_cp_feat_indices(sample, hp_cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_pos, child_word, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hp_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hp_cw_cp_dict.get((sample[s.head].pos, s.token, s.pos)):
            idx = hp_cw_cp_dict.get((sample[s.head].pos, s.token, s.pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hp_cw_cp_feat_indices_pair(head, child, hp_cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_pos, child_word, child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hp_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_idx: index of the feature (list)
    """
    if hp_cw_cp_dict.get((head.pos, child.token, child.pos)):
        return [hp_cw_cp_dict.get((head.pos, child.token, child.pos))]
    else:
        return []


def generate_hw_cw_cp_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, child_word, child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hw_cw_cp_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hw_cw_cp_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feat = (sample[s.head].token, s.token, s.pos)
            if hw_cw_cp_feat_dict.get(feat) is None:
                hw_cw_cp_feat_dict[feat] = current_idx
                current_idx += 1
    print("total (head_word, child_word, child_pos) features: ", current_idx)
    hw_cw_cp_feat_dict = OrderedDict(sorted(hw_cw_cp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hw_cw_cp.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hw_cw_cp_feat_dict, fp)
        print("saved (head_word, child_word, child_pos) features dictionary @ ", path)
    return hw_cw_cp_feat_dict


def extract_hw_cw_cp_feat_indices(sample, hw_cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, child_word, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hw_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hw_cw_cp_dict.get((sample[s.head].token, s.token, s.pos)):
            idx = hw_cw_cp_dict.get((sample[s.head].token, s.token, s.pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hw_cw_cp_feat_indices_pair(head, child, hw_cw_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, child_word, child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hw_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_idx: index of the feature (list)
    """
    if hw_cw_cp_dict.get((head.token, child.token, child.pos)):
        return [hw_cw_cp_dict.get((head.token, child.token, child.pos))]
    else:
        return []


def generate_hw_hp_cp_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, head_pos, child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hw_hp_cp_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hw_hp_cp_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feat = (sample[s.head].token, sample[s.head].pos, s.pos)
            if hw_hp_cp_feat_dict.get(feat) is None:
                hw_hp_cp_feat_dict[feat] = current_idx
                current_idx += 1
    print("total (head_word, head_pos, child_pos) features: ", current_idx)
    hw_hp_cp_feat_dict = OrderedDict(sorted(hw_hp_cp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hw_hp_cp.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hw_hp_cp_feat_dict, fp)
        print("saved (head_word, head_pos, child_pos) features dictionary @ ", path)
    return hw_hp_cp_feat_dict


def extract_hw_hp_cp_feat_indices(sample, hw_hp_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hw_hp_cp_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hw_hp_cp_dict.get((sample[s.head].token, sample[s.head].pos, s.pos)):
            idx = hw_hp_cp_dict.get((sample[s.head].token, sample[s.head].pos, s.pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hw_hp_cp_feat_indices_pair(head, child, hw_hp_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hw_hp_cp_dict: the dictionary of indices (dict)
    :return: feat_idx: index of the feature (list)
    """
    if hw_hp_cp_dict.get((head.token, head.pos, child.pos)):
        return [hw_hp_cp_dict.get((head.token, head.pos, child.pos))]
    else:
        return []


def generate_hw_hp_cw_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, head_pos, child_word)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hw_hp_cw_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hw_hp_cw_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feat = (sample[s.head].token, sample[s.head].pos, s.token)
            if hw_hp_cw_feat_dict.get(feat) is None:
                hw_hp_cw_feat_dict[feat] = current_idx
                current_idx += 1
    print("total (head_word, head_pos, child_word) features: ", current_idx)
    hw_hp_cw_feat_dict = OrderedDict(sorted(hw_hp_cw_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hw_hp_cw.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hw_hp_cw_feat_dict, fp)
        print("saved (head_word, head_pos, child_word) features dictionary @ ", path)
    return hw_hp_cw_feat_dict


def extract_hw_hp_cw_feat_indices(sample, hw_hp_cw_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_word)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hw_hp_cw_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hw_hp_cw_dict.get((sample[s.head].token, sample[s.head].pos, s.token)):
            idx = hw_hp_cw_dict.get((sample[s.head].token, sample[s.head].pos, s.token))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hw_hp_cw_feat_indices_pair(head, child, hw_hp_cw_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_word)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hw_hp_cw_dict: the dictionary of indices (dict)
    :return: feat_idx: index of the feature (list)
    """
    if hw_hp_cw_dict.get((head.token, head.pos, child.token)):
        return [hw_hp_cw_dict.get((head.token, head.pos, child.token))]
    else:
        return []


def generate_hw_cw_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, child_word)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hw_cw_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hw_cw_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feat = (sample[s.head].token, s.token)
            if hw_cw_feat_dict.get(feat) is None:
                hw_cw_feat_dict[feat] = current_idx
                current_idx += 1
    print("total (head_word, child_word) features: ", current_idx)
    hw_cw_feat_dict = OrderedDict(sorted(hw_cw_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hw_cw.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hw_cw_feat_dict, fp)
        print("saved (head_word, child_word) features dictionary @ ", path)
    return hw_cw_feat_dict


def extract_hw_cw_feat_indices(sample, hw_cw_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, child_word)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hw_cw_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hw_cw_dict.get((sample[s.head].token, s.token)):
            idx = hw_cw_dict.get((sample[s.head].token, s.token))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hw_cw_feat_indices_pair(head, child, hw_cw_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, child_word)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hw_cw_dict: the dictionary of indices (dict)
    :return: feat_idx: index of the feature (list)
    """
    if hw_cw_dict.get((head.token, child.token)):
        return [hw_cw_dict.get((head.token, child.token))]
    else:
        return []


def generate_hp_cp_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_pos, child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                    that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hp_cp_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    samp_gen = dep_sample_generator(path_to_file)
    hp_cp_feat_dict = {}
    current_idx = 0
    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            if s.token == ROOT or (word_hist.get(s.token) \
                                   and word_hist[s.token] < word_threshold):
                continue
            feat = (sample[s.head].pos, s.pos)
            if hp_cp_feat_dict.get(feat) is None:
                hp_cp_feat_dict[feat] = current_idx
                current_idx += 1
    print("total (head_pos, child_pos) features: ", current_idx)
    hp_cp_feat_dict = OrderedDict(sorted(hp_cp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + ".hp_cp.dict"
        with open(path, 'wb') as fp:
            pickle.dump(hp_cp_feat_dict, fp)
        print("saved (head_pos, child_pos) features dictionary @ ", path)
    return hp_cp_feat_dict


def extract_hp_cp_feat_indices(sample, hp_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_pos, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: hw_hp_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    feat_indices = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if hp_cp_dict.get((sample[s.head].pos, s.pos)):
            idx = hp_cp_dict.get((sample[s.head].pos, s.pos))
            if feat_indices.get(idx):
                feat_indices[idx] += 1
            else:
                feat_indices[idx] = 1
    return feat_indices


def extract_hp_cp_feat_indices_pair(head, child, hp_cp_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_pos, child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: hw_hp_cw_cp_dict: the dictionary of indices (dict)
    :return: feat_idx: index of the feature (list)
    """
    if hp_cp_dict.get((head.pos, child.pos)):
        return [hp_cp_dict.get((head.pos, child.pos))]
    else:
        return []


def generate_bigram_feat_dict(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_word, head_pos, child_word, child_pos)
    * (head_pos, child_word, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_word)
    * (head_word, child_word)
    * (head_pos, child_pos)
    :param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                                that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: bigram_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    num_features = 0
    hw_hp_cw_cp_dict = generate_hw_hp_cw_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                                      save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hw_hp_cw_cp_dict)
    hp_cw_cp_dict = generate_hp_cw_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                                save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hp_cw_cp_dict)
    hw_cw_cp_dict = generate_hw_cw_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                                save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hw_cw_cp_dict)
    hw_hp_cp_dict = generate_hw_hp_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                                save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hw_hp_cp_dict)
    hw_hp_cw_dict = generate_hw_hp_cw_feat_dict(path_to_file, word_threshold=word_threshold,
                                                save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hw_hp_cw_dict)
    hw_cw_dict = generate_hw_cw_feat_dict(path_to_file, word_threshold=word_threshold,
                                          save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hw_cw_dict)
    hp_cp_dict = generate_hp_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                          save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hp_cp_dict)
    print("total bigrams features: ", num_features)
    return hw_hp_cw_cp_dict, hp_cw_cp_dict, hw_cw_cp_dict, hw_hp_cp_dict, \
           hw_hp_cw_dict, hw_cw_dict, hp_cp_dict


def update_dict(current_dict, dict_to_add):
    """
    This function takes two dictionaries with indices as keys, and combines them.
    :param: current_dict: first dictionary
    :param: dict_to_add: second dictionary
    """
    #     comb_dict = copy.deepcopy(current_dict)
    current_num_features = len(current_dict)
    for item in dict_to_add.items():
        current_dict[current_num_features + item[0]] = item[1]


def extract_bigram_feat_indices(sample, bigram_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_word, child_pos)
    * (head_pos, child_word, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_word)
    * (head_word, child_word)
    * (head_pos, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: bigram_dict: the dictionaries of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    hw_hp_cw_cp_dict, hp_cw_cp_dict, hw_cw_cp_dict, hw_hp_cp_dict, \
    hw_hp_cw_dict, hw_cw_dict, hp_cp_dict = bigram_dict

    num_hw_hp_cw_cp_feats = len(hw_hp_cw_cp_dict)
    num_hp_cw_cp_feats = len(hp_cw_cp_dict)
    num_hw_cw_cp_feats = len(hw_cw_cp_dict)
    num_hw_hp_cp_feats = len(hw_hp_cp_dict)
    num_hw_hp_cw_feats = len(hw_hp_cw_dict)
    num_hw_cw_feats = len(hw_cw_dict)
    num_hp_cp_feats = len(hp_cp_dict)

    current_num_features = 0

    hw_hp_cw_cp_ind = extract_hw_hp_cw_cp_feat_indices(sample, hw_hp_cw_cp_dict)
    current_num_features += num_hw_hp_cw_cp_feats
    bigram_indices = copy.deepcopy(hw_hp_cw_cp_ind)

    hp_cw_cp_ind = extract_hp_cw_cp_feat_indices(sample, hp_cw_cp_dict)
    current_num_features += num_hp_cw_cp_feats
    update_dict(bigram_indices, hp_cw_cp_ind)

    hw_cw_cp_ind = extract_hw_cw_cp_feat_indices(sample, hw_cw_cp_dict)
    current_num_features += num_hw_cw_cp_feats
    update_dict(bigram_indices, hw_cw_cp_ind)

    hw_hp_cp_ind = extract_hw_hp_cp_feat_indices(sample, hw_hp_cp_dict)
    current_num_features += num_hw_hp_cp_feats
    update_dict(bigram_indices, hw_hp_cp_ind)

    hw_hp_cw_ind = extract_hw_hp_cw_feat_indices(sample, hw_hp_cw_dict)
    current_num_features += num_hw_hp_cw_feats
    update_dict(bigram_indices, hw_hp_cw_ind)

    hw_cw_ind = extract_hw_cw_feat_indices(sample, hw_cw_dict)
    current_num_features += num_hw_cw_feats
    update_dict(bigram_indices, hw_cw_ind)

    hp_cp_ind = extract_hp_cp_feat_indices(sample, hp_cp_dict)
    current_num_features += num_hp_cp_feats
    update_dict(bigram_indices, hp_cp_ind)

    return OrderedDict(sorted(bigram_indices.items(), key=lambda t: t[0]))


def extract_bigram_feat_indices_pair(head, child, bigram_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_word, head_pos, child_word, child_pos)
    * (head_pos, child_word, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_word, head_pos, child_word)
    * (head_word, child_word)
    * (head_pos, child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: bigram_dict: the dictionaries of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    hw_hp_cw_cp_dict, hp_cw_cp_dict, hw_cw_cp_dict, hw_hp_cp_dict, \
    hw_hp_cw_dict, hw_cw_dict, hp_cp_dict = bigram_dict

    num_hw_hp_cw_cp_feats = len(hw_hp_cw_cp_dict)
    num_hp_cw_cp_feats = len(hp_cw_cp_dict)
    num_hw_cw_cp_feats = len(hw_cw_cp_dict)
    num_hw_hp_cp_feats = len(hw_hp_cp_dict)
    num_hw_hp_cw_feats = len(hw_hp_cw_dict)
    num_hw_cw_feats = len(hw_cw_dict)
    num_hp_cp_feats = len(hp_cp_dict)

    current_num_features = 0

    hw_hp_cw_cp_ind = extract_hw_hp_cw_cp_feat_indices_pair(head, child, hw_hp_cw_cp_dict)
    current_num_features += num_hw_hp_cw_cp_feats
    bigram_indices = copy.deepcopy(hw_hp_cw_cp_ind)

    hp_cw_cp_ind = extract_hp_cw_cp_feat_indices_pair(head, child, hp_cw_cp_dict)
    current_num_features += num_hp_cw_cp_feats
    bigram_indices.extend(hp_cw_cp_ind)

    hw_cw_cp_ind = extract_hw_cw_cp_feat_indices_pair(head, child, hw_cw_cp_dict)
    current_num_features += num_hw_cw_cp_feats
    bigram_indices.extend(hw_cw_cp_ind)

    hw_hp_cp_ind = extract_hw_hp_cp_feat_indices_pair(head, child, hw_hp_cp_dict)
    current_num_features += num_hw_hp_cp_feats
    bigram_indices.extend(hw_hp_cp_ind)

    hw_hp_cw_ind = extract_hw_hp_cw_feat_indices_pair(head, child, hw_hp_cw_dict)
    current_num_features += num_hw_hp_cw_feats
    bigram_indices.extend(hw_hp_cw_ind)

    hw_cw_ind = extract_hw_cw_feat_indices_pair(head, child, hw_cw_dict)
    current_num_features += num_hw_cw_feats
    bigram_indices.extend(hw_cw_ind)

    hp_cp_ind = extract_hp_cp_feat_indices_pair(head, child, hp_cp_dict)
    current_num_features += num_hp_cp_feats
    bigram_indices.extend(hp_cp_ind)

    return sorted(bigram_indices)


def generate_bigram_feat_dict_minimal(path_to_file, word_threshold=0, save_to_file=False, word_hist=None):
    """
    This function generates a features dictionary, such that for every features, an index is given.
    The following features are generated for a given dataset:
    * (head_pos, child_word, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_pos, child_pos)
    igram_indices:param: path_to_file: path to location of the dataset (str)
    :param: word_threshold: if to consider a feature with word that appears less than
                that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: bigram_feat_dict: dictionary feature->index (dict)
    """
    if not word_hist:
        word_hist = generate_word_hist_dict(path_to_file)
    num_features = 0
    hp_cw_cp_dict = generate_hp_cw_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                                save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hp_cw_cp_dict)
    hw_hp_cp_dict = generate_hw_hp_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                                save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hw_hp_cp_dict)

    hp_cp_dict = generate_hp_cp_feat_dict(path_to_file, word_threshold=word_threshold,
                                          save_to_file=save_to_file, word_hist=word_hist)
    num_features += len(hp_cp_dict)
    print("total bigrams features: ", num_features)
    return hp_cw_cp_dict, hw_hp_cp_dict, hp_cp_dict


def extract_bigram_feat_indices_minimal(sample, bigram_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_pos, child_word, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_pos, child_pos)
    :param: sample: the sample to extract features from (list of DepSample)
    :param: bigram_dict: the dictionaries of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    hp_cw_cp_dict, hw_hp_cp_dict, hp_cp_dict = bigram_dict

    num_hp_cw_cp_feats = len(hp_cw_cp_dict)
    num_hw_hp_cp_feats = len(hw_hp_cp_dict)
    num_hp_cp_feats = len(hp_cp_dict)

    current_num_features = 0

    hp_cw_cp_ind = extract_hp_cw_cp_feat_indices(sample, hp_cw_cp_dict)
    current_num_features += num_hp_cw_cp_feats
    bigram_indices = copy.deepcopy(hp_cw_cp_ind)

    hw_hp_cp_ind = extract_hw_hp_cp_feat_indices(sample, hw_hp_cp_dict)
    current_num_features += num_hw_hp_cp_feats
    update_dict(bigram_indices, hw_hp_cp_ind)

    hp_cp_ind = extract_hp_cp_feat_indices(sample, hp_cp_dict)
    current_num_features += num_hp_cp_feats
    update_dict(bigram_indices, hp_cp_ind)

    return OrderedDict(sorted(bigram_indices.items(), key=lambda t: t[0]))


def extract_bigram_feat_indices_minimal_pair(head, child, bigram_dict):
    """
    This function extracts the indices (in the feature vector) of the bigrams features:
    * (head_pos, child_word, child_pos)
    * (head_word, head_pos, child_pos)
    * (head_pos, child_pos)
    :param: head: head DepSample (DepSample)
    :param: child: child DepSample (DepSample)
    :param: bigram_dict: the dictionaries of indices (dict)
    :return: feat_indices_dict: dictionary idx->count
    """
    hp_cw_cp_dict, hw_hp_cp_dict, hp_cp_dict = bigram_dict

    num_hp_cw_cp_feats = len(hp_cw_cp_dict)
    num_hw_hp_cp_feats = len(hw_hp_cp_dict)
    num_hp_cp_feats = len(hp_cp_dict)

    current_num_features = 0

    hp_cw_cp_ind = extract_hp_cw_cp_feat_indices_pair(head, child, hp_cw_cp_dict)
    current_num_features += num_hp_cw_cp_feats
    bigram_indices = copy.deepcopy(hp_cw_cp_ind)

    hw_hp_cp_ind = extract_hw_hp_cp_feat_indices_pair(head, child, hw_hp_cp_dict)
    current_num_features += num_hw_hp_cp_feats
    bigram_indices.extend(hw_hp_cp_ind)

    hp_cp_ind = extract_hp_cp_feat_indices_pair(head, child, hp_cp_dict)
    current_num_features += num_hp_cp_feats
    bigram_indices.extend(hp_cp_ind)

    return sorted(bigram_indices)