# Importing necessary libraries and loading data.
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import random
import os

# Load training and test data into pandas data frames.
df_train = pd.read_csv(os.path.join("Data", "train.csv"))
df_test = pd.read_csv(os.path.join("Data", "test.csv"))

# Dictionary to store NaN information
nans = {}


def NaN_processing(df, name, mean=True):
    """
    Replaces NaN, taking into consideration whether the data is continuous or categorical
    a) Continuous
    """
    # Add prior nulls to NaN dictionary
    nans[f"{name}_prior"] = df.isnull().sum()

    # Replaces NaN in Age, Fare with mean/median
    if mean:
        df["Age_filled"] = df["Age"].fillna(df["Age"].mean())
        df["Fare_filled"] = df["Fare"].fillna(df["Fare"].mean())
    else:
        df["Age_filled"] = df["Age"].fillna(df["Age"].median())
        df["Fare_filled"] = df["Fare"].fillna(df["Fare"].median())

    # Replace NaN in Embarked column random choice (S, C, Q)
    #df["Embarked_filled"] = df["Embarked"].fillna(random.choice(list(set(df["Embarked"]))))
    df["Embarked_filled"] = df["Embarked"].fillna("S")

    # Add post nulls to NaN dictionary
    nans[f"{name}_post"] = df.isnull().sum()


def feature_scaling(df, var):
    """
    StandardScaler from sci-kit performs fitting (estimating mean, stdev) and transformation in one step
    """
    stdsc = StandardScaler()
    return stdsc.fit_transform(np.array(df[var]).reshape(-1, 1))


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


def OHE_proccesing(df, var, drop="first"):
    """
    One-Hot-Encoding of categorical variables
    Drop ensures reduced correlation between the OneHot encoded categories
    """
    ohe = OneHotEncoder(categories="auto", drop=drop)
    return ohe.fit_transform(np.array(df[var]).reshape(-1, 1)).toarray().tolist()


def main(*dfs, verbose=False):
    """
    Input = tuple of df and name = (df, name)
    """
    for df, name in dfs:

        NaN_processing(df, name)

        df["Sex"] = df["Sex"].replace({"female": 0, "male": 1})
        df["NameLength"] = df.apply(lambda row: (len(row.Name)), axis=1)
        df["Embarked_Int"] = df.apply(lambda row: {"S": 1, "C": 2, "Q": 3}[row.Embarked_filled], axis=1)

        df["Age_StdSc"] = feature_scaling(df, "Age_filled")
        df["Fare_StdSc"] = feature_scaling(df, "Fare_filled")
        df["Name_StdSc"] = feature_scaling(df, "NameLength")
        df["SibSp_StdSc"] = feature_scaling(df, "SibSp")
        df["Parch_StdSc"] = feature_scaling(df, "Parch")
        df["Pclass_StdSc"] = feature_scaling(df, "Pclass")
        df["Embarked_StdSc"] = feature_scaling(df, "Embarked_Int")

        df["Age_Class"] = df.apply(lambda row: Age_Classification(row.Age), axis=1)
        df["SibSp_Class"] = df.apply(lambda row: SibSp_classification(row.SibSp), axis=1)
        df["Parch_Class"] = df.apply(lambda row: Parch_classification(row.Parch), axis=1)

        df["AgeOHE"] = OHE_proccesing(df, "Age_Class")
        df["SibSpOHE"] = OHE_proccesing(df, "SibSp_Class")
        df["ParchOHE"] = OHE_proccesing(df, "Parch_Class")
        df["EmbarkedOHE"] = OHE_proccesing(df, "Embarked_filled")

        if verbose:
            df.to_csv(os.path.join("Data", f"{name}_processed.csv"), na_rep="NaN")


main((df_train, "train"), (df_test, "test"), verbose=False)

