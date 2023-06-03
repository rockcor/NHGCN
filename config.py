import torch
import os
import numpy as np
import random


class Config:
    # num_training = 1208
    # num_testing = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
