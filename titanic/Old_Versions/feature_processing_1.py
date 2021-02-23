# Importing necessary libraries and loading data.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

sns.set(color_codes=True)

# Load training and test data into pandas data frames.
df_train = pd.read_csv("../Data/train.csv")
df_train_org = pd.read_csv("../Data/train.csv")
df_test = pd.read_csv("../Data/test.csv")
df_test_org = pd.read_csv("../Data/test.csv")

def NaN_processing(df):
    df["Age"] = df["Age"].fillna(df["Age"].mean())      # Replace NaN in Age column with mean (or median)
    df["Embarked"] = df["Embarked"].fillna("X")         # Replace NaN in Embarked column with X
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())   # Replace NaN in Fare column with mean (or median)


# NaN processing of test data
train_nulls_prior = df_train.isnull().sum()
NaN_processing(df_train)
train_nulls_post = df_train.isnull().sum()

# NaN processing of train data
test_nulls_prior = df_train.isnull().sum()
NaN_processing(df_test)
test_nulls_post = df_train.isnull().sum()


def replace_by_map(df, var, map):
    df[var] = df[var].replace(map)


# Replacing sex with numerical values (0 if female else 1) in train and test set.
replace_by_map(df_train, "Sex", {"female": 0, "male": 1})
replace_by_map(df_test, "Sex", {"female": 0, "male": 1})


def feature_scaling(df, var):
    stdsc = StandardScaler()
    return stdsc.fit_transform(np.array(df[var]).reshape(-1, 1))


df_train["Age_StdSc"] = feature_scaling(df_train, "Age")
df_train["Fare_StdSc"] = feature_scaling(df_train, "Fare")
df_test["Age_StdSc"] = feature_scaling(df_test, "Age")
df_test["Fare_StdSc"] = feature_scaling(df_test, "Fare")


# Pclass scaling
mapping = {1: 0, 2: 0.5, 3: 1}
df_train["Pclass_Scale"] = df_train.apply(lambda row: mapping[row.Pclass], axis=1)
df_test["Pclass_Scale"] = df_test.apply(lambda row: mapping[row.Pclass], axis=1)


def SibSp_classification(number):
    if number == 0:
        return 0
    elif number == 1:
        return 1
    elif number == 2:
        return 2
    else:
        return 3

def Parch_classification(number):
    if number == 0:
        return 0
    elif number == 1:
        return 1
    elif number == 2:
        return 2
    elif number == 3:
        return 3
    else:
        return 4

def Age_Classification(number):
    """
    Based on the intervals defined in classification.py
    """
    if pd.isnull(number):
        return "Unknown"
    elif number < 9:
        return "Child"
    elif number < 18:
        return "Teenager"
    elif number < 22:
        return "YoungAdult"
    elif number < 55:
        return "Adult"
    else:
        return "Eldery"


df_train["Age_Class"] = df_train_org.apply(lambda row: Age_Classification(row.Age), axis=1)
df_test["Age_Class"] = df_test_org.apply(lambda row: Age_Classification(row.Age), axis=1)

df_train["SibSp_Class"] = df_train.apply(lambda row: SibSp_classification(row.SibSp), axis=1)
df_test["SibSp_Class"] = df_test.apply(lambda row: SibSp_classification(row.SibSp), axis=1)

df_train["Parch_Class"] = df_train.apply(lambda row: Parch_classification(row.Parch), axis=1)
df_test["Parch_Class"] = df_test.apply(lambda row: Parch_classification(row.Parch), axis=1)

def OHE_proccesing(df, var):
    # Drop ensures reduced correlation between the OneHot encoded categories
    embark_ohe = OneHotEncoder(categories="auto", drop="first")
    return embark_ohe.fit_transform(np.array(df[var]).reshape(-1, 1)).toarray().tolist()


df_train["AgeOHE"] = OHE_proccesing(df_train, "Age_Class")
df_test["AgeOHE"] = OHE_proccesing(df_test, "Age_Class")

df_train["SibSpOHE"] = OHE_proccesing(df_train, "SibSp_Class")
df_test["SipSpOHE"] = OHE_proccesing(df_test, "SibSp_Class")

df_train["ParchOHE"] = OHE_proccesing(df_train, "Parch_Class")
df_test["ParchOHE"] = OHE_proccesing(df_test, "Parch_Class")

df_train["EmbarkedOHE"] = OHE_proccesing(df_train, "Embarked")
df_test["EmbarkedOHE"] = OHE_proccesing(df_test, "Embarked")