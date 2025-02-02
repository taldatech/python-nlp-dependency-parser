import time
import os
import pickle
import numpy as np
from collections import OrderedDict
from collections import namedtuple
from random import shuffle
import copy
from itertools import combinations, combinations_with_replacement
from utils.utils import ROOT, dep_sample_generator
from typing import List, Dict
from utils.utils import DepSample


def generate_inbetween_features(sample, s):
    """
    This function generates the features:
    (head word, b word, child word) for h.idx < b.idx < c.idx
    :param: sample (list of DepSamples)
    :param: s (DepSample)
    :return: inbetween_features (list of tuples)
    """
    start_idx = min(s.idx, sample[s.head].idx)
    end_idx = max(s.idx, sample[s.head].idx)
    inbetween_features = []
    for i in range(start_idx + 1, end_idx):
        inbetween_features.append((sample[s.head].pos, sample[i].pos, s.pos))
    return inbetween_features


def generate_features_dict(path_to_file, sample2features, start_idx, feature_threshold=0, save_to_file=False,
                           features_name=''):
    """
    This function generates a features dictionary, such that for every features, an index is given starting from start
    the given start_idx.
    The following features are generated for a given dataset:
    byt the given sample2features which is a lambda exprestion to extract a feature from a sample
    you can think of sample2features as a template
    :param: path_to_file: path to location of the dataset (str)
    :param: feature_threshold: if to consider a feature with word that appears less than that in the dataset (int)
    :param: save_to_file: whether or not to save the dictionary on the disk (bool)
    :param: word_hist: dictionary of words histogram in the dataset (dict)
    :return: hw_hp_feat_dict: dictionary feature->index (dict)
    """

    samp_gen = dep_sample_generator(path_to_file)
    hw_hp_feat_dict = OrderedDict()
    features_hist = OrderedDict()

    for s_i, sample in enumerate(samp_gen):
        for s in sample:
            # ignore ROOT
            if s.token == ROOT:
                continue

            feats = sample2features(sample, s)
            for feat in feats:
                features_hist[feat] = features_hist.get(feat, 0) + 1

    current_idx = start_idx
    for k, v in features_hist.items():
        if v > feature_threshold:
            hw_hp_feat_dict[k] = current_idx
            current_idx += 1

    # print("total {:} features: ".format(features_name), current_idx)
    hw_hp_feat_dict = OrderedDict(sorted(hw_hp_feat_dict.items(), key=lambda t: t[1]))
    if save_to_file:
        path = path_to_file + features_name + '.dict'
        with open(path, 'wb') as fp:
            pickle.dump(hw_hp_feat_dict, fp)
        print("saved {:} features dictionary @ ".format(features_name), path)
    return hw_hp_feat_dict


def generate_features_dicts(path_to_file: str, save_to_file: bool = False, minimal: bool = False
                            , use_mcdonald: bool = False, feature_threshold: int = 0) -> Dict[str, dict]:
    """
    given a training file we return a dictionary of dictionaries
    where key is feature type name, and value is a dictionary of that feature generated by 'generate_features_dict'
    according to feature templates required by hw2 pg 2, those templates are:
    head word _ head pos
    head word
    head pos
    child word _ child pos
    child word
    child pos
    h_pos c_word c_pos

    if not minimal we add:
        h_word h_pos c_word c_pos
        h_word c_word c_pos
        h_word h_pos c_word
        h_word c_word

    if use mcdonald:
        distance(head,child)
        # is head to the right or left of the child:
        1 if head.idx < child.idx
        # in-between POS features:
        (h_pos, b_pos, c_pos)
        # surrounding word POS features:
        (h_pos, h+1_pos, c-1_pos, c_pos)
        (h-1_pos, h_pos, c-1_pos, c_pos)
        (h_pos, h+1_pos, c_pos, c+1_pos)
        (h-1_pos, h_pos, c_pos, c+1_pos)
    :param path_to_file: training file to extract features from
    :param save_to_file: if to save the dictionary
    :param minimal: if to add the extra features as described
    :param use_mcdonald: use the McDonald's paper features
    :param feature_threshold: keep features thast appear more than this
    :return: dictionary described above
    """

    feature_types = {'head word _ head pos': (lambda sample, s: [(sample[s.head].token, sample[s.head].pos)],
                                              feature_threshold),
                     'head word': (lambda sample, s: [(sample[s.head].token)], feature_threshold),
                     'head pos': (lambda sample, s: [(sample[s.head].pos)], feature_threshold),
                     'child word _ child pos': (lambda sample, s: [(s.token, s.pos)], feature_threshold),
                     'child word': (lambda sample, s: [(s.token)], feature_threshold),
                     'child pos': (lambda sample, s: [(s.pos)], feature_threshold),
                     'h_pos c_word c_pos': (lambda sample, s: [(sample[s.head].pos, s.token, s.pos)],
                                            feature_threshold),
                     'h_word h_pos c_pos': (lambda sample, s: [(sample[s.head].token, sample[s.head].pos, s.pos)],
                                            feature_threshold),
                     'h_pos c_pos': (lambda sample, s: [(sample[s.head].pos, s.pos)], feature_threshold),
                     }

    if not minimal:
        feature_types['h_word h_pos c_word c_pos'] = (lambda sample, s:
                                                      [(sample[s.head].token, sample[s.head].pos, s.token, s.pos)],
                                                      feature_threshold)

        feature_types['h_word c_word c_pos'] = (lambda sample, s:
                                                [(sample[s.head].token, s.token, s.pos)], feature_threshold)

        feature_types['h_word h_pos c_word'] = (lambda sample, s:
                                                [(sample[s.head].token, sample[s.head].pos, s.token)],
                                                feature_threshold)

        feature_types['h_word c_word'] = (lambda sample, s:
                                          [(sample[s.head].token, s.token)], feature_threshold)

    if use_mcdonald:
        # distance + is head to the right of child? (-) if child > head, (+) else
        # maybe we can add the distance to the unigrams?
        feature_types['h_word c_word dist'] = (lambda sample, s:
                                               [(sample[s.head].token, s.token, sample[s.head].idx - s.idx)],
                                               feature_threshold)

        feature_types['h_word c_word direction'] = (lambda sample, s:
                                                    [(sample[s.head].token, s.token,
                                                      np.sign(sample[s.head].idx - s.idx))],
                                                    feature_threshold)

        feature_types['h_pos c_pos direction'] = (lambda sample, s:
                                                  [(sample[s.head].pos, s.pos,
                                                    np.sign(sample[s.head].idx - s.idx))],
                                                  feature_threshold)
        # in-between POS features:
        feature_types['h_c_pos_seq'] = (lambda sample, s:
                                        [tuple(l.pos for l in sample[sample[s.head].idx: s.idx + 1])],
                                        feature_threshold)

        feature_types['h_b_c_pos'] = (lambda sample, s: generate_inbetween_features(sample, s), feature_threshold)
        # surrounding word POS features
        feature_types['h_pos h_next_pos c_prev_pos c_pos'] = (lambda sample, s:
                                                              [(sample[s.head].pos,
                                                                sample[min(s.head + 1, sample[-1].idx)].pos,
                                                                sample[max(s.idx - 1, 0)].pos,
                                                                s.pos)], 2)

        feature_types['h_prev_pos h_pos c_prev_pos c_pos'] = (lambda sample, s:
                                                              [(sample[max(s.head - 1, 0)].pos,
                                                                sample[s.head].pos,
                                                                sample[max(s.idx - 1, 0)].pos,
                                                                s.pos)], feature_threshold)

        feature_types['h_pos h_next_pos c_pos c_next_pos'] = (lambda sample, s:
                                                              [(sample[s.head].pos,
                                                                sample[min(s.head + 1, sample[-1].idx)].pos,
                                                                s.pos,
                                                                sample[min(s.idx + 1, sample[-1].idx)].pos)],
                                                              feature_threshold)

        feature_types['h_prev_pos h_pos c_pos c_next_pos'] = (lambda sample, s:
                                                              [(sample[max(s.head - 1, 0)].pos,
                                                                sample[s.head].pos,
                                                                s.pos,
                                                                sample[min(s.idx + 1, sample[-1].idx)].pos)],
                                                              feature_threshold)
    features_dicts = {}
    current_num_features = 0
    for feature_type_name, (feature_template, feature_threshold) in feature_types.items():
        features_dicts[feature_type_name] = generate_features_dict(path_to_file,
                                                                   feature_template,
                                                                   start_idx=current_num_features,
                                                                   feature_threshold=feature_threshold,
                                                                   save_to_file=False,
                                                                   features_name=feature_type_name)

        num_features = len(features_dicts[feature_type_name])
        current_num_features += num_features

        print('generated {:} features , num features: {:}, total num features: {:}'.format(feature_type_name,
                                                                                           num_features,
                                                                                           current_num_features))
        if save_to_file:
            path = path_to_file + ".feat_dicts"
            with open(path, 'wb') as fp:
                pickle.dump(features_dicts, fp)
            print("saved features dictionaries @ ", path)

    return features_dicts


def extract_local_feature_indices(head: DepSample, child: DepSample, dictionaries: Dict[str, OrderedDict],
                                  minimal: bool, use_mcdonald: bool = False, sample=None):
    """
    returns a list of indices that turned on given the head location
    :param head: head DepSAmple
    :param child: child/ current DepSample
    :param sample: list of DepSamples of the complete sentence (for mcdonalds features)
    :param dictionaries:  dictionaries of features as generated by 'generate_features_dicts'
    :param minimal: if we're operating on minimal or full model
    :param use_mcdonald: whether or not to use features from McDonald's paper
    :return: list of features that turned on on the given head and  child
    """

    # dictionary_names = [
    #     'head word _ head pos',
    #     'head word',
    #     'head pos',
    #     'child word _ child pos',
    #     'child word',
    #     'child pos',
    #     'h_pos c_word c_pos',
    #     'h_word h_pos c_pos',
    #     'h_pos c_pos',
    #     'h_word h_pos c_word c_pos',
    #     'h_word c_word c_pos',
    #     'h_word h_pos c_word',
    #     'h_word c_word']

    feature_ind = []
    idx = dictionaries['head word _ head pos'].get((head.token, head.pos))
    if idx is not None:
        feature_ind.append(idx)

    idx = dictionaries['head word'].get((head.token))
    if idx is not None:
        feature_ind.append(idx)

    idx = dictionaries['head pos'].get((head.pos))
    if idx is not None:
        feature_ind.append(idx)

    idx = dictionaries['child word _ child pos'].get((child.token, child.pos))
    if idx is not None:
        feature_ind.append(idx)

        idx = dictionaries['child word'].get((child.token))
    if idx is not None:
        feature_ind.append(idx)

    idx = dictionaries['child pos'].get((child.pos))
    if idx is not None:
        feature_ind.append(idx)

    idx = dictionaries['h_pos c_word c_pos'].get((head.pos, child.token, child.pos))
    if idx is not None:
        feature_ind.append(idx)

    idx = dictionaries['h_word h_pos c_pos'].get((head.token, head.pos, child.pos))
    if idx is not None:
        feature_ind.append(idx)

    idx = dictionaries['h_pos c_pos'].get((head.pos, child.pos))
    if idx is not None:
        feature_ind.append(idx)

    if not minimal:
        # [h_word h_pos c_word c_pos,
        # 'h_word c_word c_pos',
        # 'h_word h_pos c_word',
        # 'h_word c_word']

        idx = dictionaries['h_word h_pos c_word c_pos'].get((head.token, head.pos, child.token, child.pos))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_word c_word c_pos'].get((head.token, child.token, child.pos))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_word h_pos c_word'].get((head.token, head.pos, child.token))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_word c_word'].get((head.token, child.token))
        if idx is not None:
            feature_ind.append(idx)

    if use_mcdonald and sample is not None:
        idx = dictionaries['h_word c_word dist'].get((head.token, child.token, head.idx - child.idx))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_word c_word direction'].get((head.token, child.token, np.sign(head.idx - child.idx)))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_pos c_pos direction'].get((head.pos, child.pos, np.sign(head.idx - child.idx)))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_c_pos_seq'].get(tuple(l.pos for l in sample[head.idx: child.idx + 1]))
        if idx is not None:
            feature_ind.append(idx)

        start_idx = min(child.idx, head.idx)
        end_idx = max(child.idx, head.idx)
        for i in range(start_idx + 1, end_idx):
            idx = dictionaries['h_b_c_pos'].get((head.pos, sample[i].pos, child.pos))
            if idx is not None:
                feature_ind.append(idx)

        idx = dictionaries['h_pos h_next_pos c_prev_pos c_pos'].get((sample[head.idx].pos,
                                                                     sample[min(head.idx + 1, sample[-1].idx)].pos,
                                                                     sample[max(child.idx - 1, 0)].pos,
                                                                     child.pos))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_prev_pos h_pos c_prev_pos c_pos'].get((sample[max(head.idx - 1, 0)].pos,
                                                                     sample[head.idx].pos,
                                                                     sample[max(child.idx - 1, 0)].pos,
                                                                     child.pos))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_pos h_next_pos c_pos c_next_pos'].get((sample[head.idx].pos,
                                                                     sample[min(head.idx + 1, sample[-1].idx)].pos,
                                                                     child.pos,
                                                                     sample[min(child.idx + 1, sample[-1].idx)].pos))
        if idx is not None:
            feature_ind.append(idx)

        idx = dictionaries['h_prev_pos h_pos c_pos c_next_pos'].get((sample[max(head.idx - 1, 0)].pos,
                                                                     sample[head.idx].pos,
                                                                     child.pos,
                                                                     sample[min(child.idx + 1, sample[-1].idx)].pos))
        if idx is not None:
            feature_ind.append(idx)

    return feature_ind


def extract_global_features(sample: List[DepSample], dictionaries: Dict[str, OrderedDict], minimal: bool,
                            use_mcdonald: bool = False) \
        -> Dict[int, int]:
    """
    given a sample we return a dictionary where keys are features that turned on
    and values are how manny times they turned on
    :param sample: list of DepSample to return features of
    :param dictionaries: of features generated by 'generate_features_dicts'
    :param minimal: if the model is with minimal features
    :param use_mcdonald: use the McDonald's paper features
    :return: dict
    """

    features_counters = {}

    for s in sample:

        if s.token == ROOT:
            continue

        feature_ind = extract_local_feature_indices(sample[s.head], s, dictionaries, minimal, use_mcdonald, sample)
        for feat in feature_ind:
            features_counters[feat] = features_counters.get(feat, 0) + 1

    return features_counters


def generate_global_features_dict(training_file_name: str, feature_extractor, dicts,
                                  save_to_file: bool, minimal=True, use_mcdonald=True) -> dict:
    """
    returns a dictionary where key is sentence index and value is a dict features of the sentence
    this method exists only as a way to help speed up training by pre calculating all needed fc graphs
    :param training_file_name: to extract all possible sentence lengths seen in the data
    :param: feature_extractor: function to extract feature for (head, child)
    :param: dicts: dictionaries of features (list of dicts)
    :param save_to_file: if to save generated dict to a pickle
    :param: minimal: whether or not to use the minimal version of the features
    :param use_mcdonald: use the McDonald's paper features
    :return: the described dict of dicts
    """
    path = training_file_name + ".gt_global_features.dict"
    if os.path.isfile(path):
        with open(path, 'rb') as fp:
            global_features_dict = pickle.load(fp)
            print("loaded global samples to features dictionary from ", path)
            return global_features_dict

    print("generating features for all samples in the training set")
    st_time = time.time()
    global_features_dict = {}
    for idx, sample in enumerate(dep_sample_generator(training_file_name)):
        global_features_dict[idx] = feature_extractor(sample, dicts, minimal, use_mcdonald)

    if save_to_file:
        with open(path, 'wb') as fp:
            pickle.dump(global_features_dict, fp)
        print("saved ground truth global features dictionary @ ", path)
    print("time took: %.3f secs" % (time.time() - st_time))
    return global_features_dict
