from scipy.stats import ks_2samp

import featuretools as ft
# Import libraries and set desired options
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

from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

import  matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import lightgbm as lgb
import seaborn as sns
from collections import Counter


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
plt.ion()


#-----------------------------------------------------------------
def add_feature_to_sparse(spars_matrxi, feature_array_or_matrix):
    return hstack([spars_matrxi, feature_array_or_matrix])

def check_lr_crossval(sparse_matrix, y, n_splits=2, balanced=True):
    time_split = TimeSeriesSplit(n_splits=n_splits)
    if balanced:
        logit = LogisticRegression(C=1, random_state=17, class_weight='balanced')
    else:
        logit = LogisticRegression(C=1, random_state=17)
    
    cv_scores = cross_val_score(logit, sparse_matrix, y, cv=time_split, scoring='roc_auc', n_jobs=-1) 
    print(cv_scores, cv_scores.mean(), cv_scores.std())
    return cv_scores, cv_scores.mean(), cv_scores.std()
#------------------------------------------------------------------

# cv = CountVectorizer(ngram_range=(1, 3), max_features=50000)
# with open('train_sessions_text.txt') as inp_train_file:
#     X_train = cv.fit_transform(inp_train_file)
# with open('test_sessions_text.txt') as inp_test_file:
#     X_test = cv.transform(inp_test_file)
    
# print(f'Original size: {X_train.shape}, {X_test.shape}')

# X_train_new = add_time_features(train_df.fillna(0), X_train)
# X_test_new = add_time_features(test_df.fillna(0), X_test)                         
# print(f'After add base features: {X_train_new.shape}, {X_test_new.shape}')
# cat_names = ['part_of_day', 'Month', 'Week', 'Day', 'Dayofweek']

#--------------------------------------------------------------------------------
# cont_names = ['n_sites_in_session', 'n_uniq_sites_in_session', 'n_dupl_sites_in_session', 
#                'mean_sites_duration', 'sess_duration']

#------------------------------------------------------------------



#------------------------------------------------------------------
#      PLEASE PAY ATTENTION, GLOABAL VASRS DECLAIRED IN THE END
#------------------------------------------------------------------

def process_sites():
    train_sites = train_df[sites].fillna(0).astype(np.int32)
    test_sites = test_df[sites].fillna(0).astype(np.int32)
    return train_sites, test_sites


#-------------------------------------------------------------------
def load_dataframes():
    train_df = pd.read_csv('../data/train_sessions.csv', index_col='session_id')
    test_df = pd.read_csv('../data/test_sessions.csv', index_col='session_id')

    # Convert time1, ..., time10 columns to datetime type
    times = ['time%s' % i for i in range(1, 11)]
    train_df[times] = train_df[times].apply(pd.to_datetime)
    test_df[times] = test_df[times].apply(pd.to_datetime)

    # Sort the data by time
    train_df = train_df.sort_values(by='time1')

    # Look at the first rows of the training set
    return train_df, test_df


#-------------------------------------------------------------------
# A helper function for writing predictions to a file
def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
    
    
#---------------------------------------------------------------------


SEED=13
device = 'cuda' if torch.cuda.is_available() else 'cpu'



