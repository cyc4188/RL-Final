import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import random
import math
from collections import deque, namedtuple
import time
import gym
import pickle

def save_file(game_name, file_name, data):
    """ save data to file """
    with open('./{}/{}'.format(game_name, file_name), "wb") as f:
        pickle.dump(data, f)

def get_file(game_name, file_name):
    """ get data from file """
    with open('./{}/{}'.format(game_name, file_name), "rb") as f:
        data = pickle.load(f)
    return data


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
