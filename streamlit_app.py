import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier

#
from sklearn.linear_model import SGDClassifier

#
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

#categorical data to number
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Train test split
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score;
loan_data = pd.read_csv("loan_approval_dataset.csv")
