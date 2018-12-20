from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import featuretools as ft
import numpy as np
import pandas as pd
import gc, re, torch, warnings
import pickle as pickle
import scipy
import os
from pathlib import Path
from fastai import *
from fastai.collab import *
from fastai.tabular import *
pd.options.display.max_rows = 460

# from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

import  matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import lightgbm as lgb
import seaborn as sns
from collections import Counter
from scipy.special import erfinv

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
plt.ion()

SEED=13
device = 'cuda' if torch.cuda.is_available() else 'cpu'



