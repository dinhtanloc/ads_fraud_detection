import datetime
import enum
import itertools
import math
import os
import pickle
import random as rd
import re
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import shap
import spacy
import tensorflow as tf
import xgboost as xgb
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras import backend as K, layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from lightgbm import LGBMClassifier
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.offline import iplot, plot
from plotly.subplots import make_subplots
from scipy import sparse, stats
from scipy.sparse import csr_matrix, hstack
from scipy.stats import boxcox, norm, uniform, yeojohnson
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import (
    RFE, SelectFromModel, SelectKBest, f_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, roc_curve
)
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold,
    train_test_split, cross_val_score, learning_curve
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, LabelEncoder, MinMaxScaler, OneHotEncoder,
    QuantileTransformer, StandardScaler
)
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from spacy.lang.en.stop_words import STOP_WORDS
from tabnet_keras import TabNetClassifier, TabNetRegressor
from tensorflow.keras import layers as Layers
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from wordcloud import WordCloud
import lime
import wandb