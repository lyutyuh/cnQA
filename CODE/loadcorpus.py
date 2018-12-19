import torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json
import numpy as np
import pickle
from tqdm import tqdm

def load_corpus():
    with open('../DATA/train_answ_id_vec.data') as fin:
        x_train_ans = eval(fin.read())
    with open('../DATA/train_answ_pos_tag.data') as fin:
        x_train_ans_pos = eval(fin.read())
    with open('../DATA/train_prob_id_vec.data') as fin:
        x_train_que = eval(fin.read())
    with open('../DATA/train_prob_pos_tag.data') as fin:
        x_train_que_pos = eval(fin.read())
    with open('../DATA/train_answ_lap.data') as fin:
        x_train_ans_overlap = eval(fin.read())
    with open('../DATA/train_prob_lap.data') as fin:
        x_train_que_overlap = eval(fin.read())
    with open('../DATA/train_label.data') as fin:    
        y_train = eval(fin.read())


    with open('../DATA/valid_answ_id_vec.data') as fin:
        x_valid_ans = eval(fin.read())
    with open('../DATA/valid_answ_pos_tag.data') as fin:
        x_valid_ans_pos = eval(fin.read())
    with open('../DATA/valid_prob_id_vec.data') as fin:
        x_valid_que = eval(fin.read())
    with open('../DATA/valid_prob_pos_tag.data') as fin:
        x_valid_que_pos = eval(fin.read())
    with open('../DATA/valid_answ_lap.data') as fin:
        x_valid_ans_overlap = eval(fin.read())
    with open('../DATA/valid_prob_lap.data') as fin:
        x_valid_que_overlap = eval(fin.read())
    with open('../DATA/valid_label.data') as fin:    
        y_valid = eval(fin.read())
    print("In training set: ans,que,label", len(x_train_ans), len(x_train_ans_pos), len(x_train_que), len(x_train_que_pos), len(y_train))
    print("In validation set: ans,que,label:",len(x_valid_ans), len(x_valid_ans_pos), len(x_valid_que), len(x_valid_que_pos), len(y_valid))
    return x_train_ans, x_train_ans_pos, x_train_que, x_train_que_pos, x_train_ans_overlap, x_train_que_overlap, y_train, x_valid_ans, x_valid_ans_pos, x_valid_que, x_valid_que_pos, x_valid_ans_overlap, x_valid_que_overlap, y_valid

def load_vocab():
    with open('../DATA/vocabulary.data') as fin:
        vocab = eval(fin.read())
    return vocab