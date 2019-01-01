import os
import pickle
import numpy as np
from collections import OrderedDict
from collections import namedtuple

import copy

# named tuple has methods like _asdict()
ROOT = "*"
DepSample = namedtuple('DepSample', 'idx, token, pos, head')


def split_train_validation(path_to_file, valid_amount=0.2):
    """
    This functions takes a train dataset and splits it to trainining
    and validation sets accoording to `valid_amount`.
    :param: path_to_file: path to file containing the dataset (str)
    :param: valid_amount: percentage of samples to take for validation (float)
    :return: train_file_path: path to file containing the training samples (str)
    :return: valid_file_path: path to file containing the validation samples (str)
    """
    path_train_file = path_to_file + ".train.labeled"
    path_valid_file = path_to_file + ".valid.labeled"
    # count samples
    samp_gen = dep_sample_generator(path_to_file)
    total_samples = 0
    for _ in samp_gen:
        total_samples += 1
    print("total samples ", total_samples)
    buffer = []
    num_validation = int(valid_amount * total_samples)
    num_training = total_samples - num_validation
    taken_for_training = 0
    t_file = open(path_train_file, 'w')
    v_file = open(path_valid_file, 'w')
    with open(path_to_file) as fp:
        sample = []
        for line in fp:
            if not line.rstrip():
                if taken_for_training < num_training:
                    for l in sample:
                        t_file.write(l)
                    t_file.write('\n')
                    taken_for_training += 1
                else:
                    for l in sample:
                        v_file.write(l)
                    v_file.write('\n')
                sample = []
            else:
                sample.append(line)

        if taken_for_training < num_training:
            for l in sample:
                t_file.write(l)
            t_file.write('\n')
            taken_for_training += 1
        else:
            for l in sample:
                v_file.write(l)
            v_file.write('\n')
    t_file.close()
    v_file.close()
    print("num training: ", num_training, " saved @ ", path_train_file)
    print("num validation: ", num_validation, " saved @ ", path_valid_file)


def dep_sample_generator(path_to_file):
    """
    This function generates samples, such that every sample is a list
    ordered by the tokens' counter (first column).
    :param: path_to_file: string to the location of the file tor read from (str)
    :return: sample (list of DepSample)
    """
    assert os.path.isfile(path_to_file), "File does not exist"
    root = DepSample(0, ROOT, ROOT, 0)
    with open(path_to_file) as fp:
        sample = [root]
        for line in fp:
            if not line.rstrip():
                yield sample
                sample = [root]
            else:
                ls = line.rstrip().split('\t')
                #                 print(ls)
                sample.append(DepSample(int(ls[0]), ls[1], ls[3], int(ls[6])))
        if len(sample) > 1:
            yield sample


def sample_to_successors(sample):
    """
    This function converts sample representation in the form of list of DepSample to
    the form of Graph successors: map between heads to list of childs.
    :param: sample: the original sample (list of DepSample)
    :return: succ_rep: dictionary head->list_of_children (dict)
    """
    succ_rep = {}
    for s in sample:
        if s.token == ROOT:
            continue
        if succ_rep.get(s.head) is not None:
            succ_rep[s.head].append(s.idx)
        else:
            succ_rep[s.head] = [s.idx]
    return succ_rep


def sample_to_full_successors(N):
    """
    This function converts sample representation in the form of list of DepSample to
    the form of FULLY CONNECTED Graph successors: map between heads to list of childs.
    :param: N: length of the sentence
    :return: succ_rep: dictionary head->list_of_children (dict)
    """
    succ_rep = {}
    nodes_ids = list(range(N + 1))
    for i in nodes_ids:
        new_node_ids = copy.deepcopy(nodes_ids)
        new_node_ids.remove(0)
        if i > 0:
            new_node_ids.remove(i)
        succ_rep[i] = new_node_ids
    return succ_rep


def successors_to_sample(sample_no_head, succ_rep):
    """
    This function converts successors representation to list of DepSample.
    :param: sample_no_head: list of DepSample where s.head=None
    :param: succ_rep: dictionary head->list_of_children (dict)
    :return: sample_with_head (list of DepSamples)
    """
    root = DepSample(0, ROOT, ROOT, 0)
    sample_with_heads = [root]
    for head in succ_rep.keys():
        childs = succ_rep[head]
        for c in childs:
            new_sample = DepSample(sample_no_head[c].idx, sample_no_head[c].token,
                                   sample_no_head[c].pos, head)
            sample_with_heads.append(new_sample)
    return sorted(sample_with_heads, key=lambda t: t.idx)
