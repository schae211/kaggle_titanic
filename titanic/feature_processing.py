# Importing necessary libraries and loading data.
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def load_and_process_data_set():
    # Load training and test data into pandas data frames.
    df_train = pd.read_csv("Data/train.csv")
    df_train_org = pd.read_csv("Data/train.csv")
    df_test = pd.read_csv("Data/test.csv")
    df_test_org = pd.read_csv("Data/test.csv")

    # NaN processing of test data
    NaN_processing(df_train)

    # NaN processing of train data
    NaN_processing(df_test)

    # Pclass scaling
    mapping = {1: 0, 2: 0.5, 3: 1}
    df_train["Pclass_Scale"] = df_train.apply(lambda row: mapping[row.Pclass], axis=1)
    df_test["Pclass_Scale"] = df_test.apply(lambda row: mapping[row.Pclass], axis=1)

    df_train["NameLength"] = df_train.apply(lambda row: (len(row.Name)), axis=1)
    df_test["NameLength"] = df_test.apply(lambda row: (len(row.Name)), axis=1)

    # Replacing sex with numerical values (0 if female else 1) in train and test set.
    replace_by_map(df_train, "Sex", {"female": 0, "male": 1})
    replace_by_map(df_test, "Sex", {"female": 0, "male": 1})

    df_train["Age_StdSc"], df_test["Age_StdSc"] = feature_scaling(df_train, df_test, "Age")
    df_train["Fare_StdSc"], df_test["Fare_StdSc"] = feature_scaling(df_train, df_test, "Fare")
    df_train["Name_StdSc"], df_test["Name_StdSc"] = feature_scaling(df_train, df_test, "NameLength")

    df_train["Age_Class"] = df_train_org.apply(lambda row: Age_Classification(row.Age), axis=1)
    df_test["Age_Class"] = df_test_org.apply(lambda row: Age_Classification(row.Age), axis=1)

    df_train["SibSp_Class"] = df_train.apply(lambda row: SibSp_classification(row.SibSp), axis=1)
    df_test["SibSp_Class"] = df_test.apply(lambda row: SibSp_classification(row.SibSp), axis=1)

    df_train["Parch_Class"] = df_train.apply(lambda row: Parch_classification(row.Parch), axis=1)
    df_test["Parch_Class"] = df_test.apply(lambda row: Parch_classification(row.Parch), axis=1)

    df_train["AgeOHE"] = OHE_proccesing(df_train, "Age_Class")
    df_test["AgeOHE"] = OHE_proccesing(df_test, "Age_Class")

    df_train["SibSpOHE"] = OHE_proccesing(df_train, "SibSp_Class")
    df_test["SipSpOHE"] = OHE_proccesing(df_test, "SibSp_Class")

    df_train["ParchOHE"] = OHE_proccesing(df_train, "Parch_Class")
    df_test["ParchOHE"] = OHE_proccesing(df_test, "Parch_Class")

    df_train["EmbarkedOHE"] = OHE_proccesing(df_train, "Embarked")
    df_test["EmbarkedOHE"] = OHE_proccesing(df_test, "Embarked")
    return df_train, df_test


def NaN_processing(df):
    df["Age"] = df["Age"].fillna(df["Age"].mean())      # Replace NaN in Age column with mean (or median)
    df["Embarked"] = df["Embarked"].fillna("X")         # Replace NaN in Embarked column with X
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())   # Replace NaN in Fare column with mean (or median)


def replace_by_map(df, var, map):
    df[var] = df[var].replace(map)


def feature_scaling(df_train, df_test, var):
    stdsc = StandardScaler()
    values = np.array(df_train[var])
    values = np.append(values, df_test[var])
    stdsc.fit(values.reshape(-1, 1))
    return stdsc.transform(np.array(df_train[var]).reshape(-1, 1)), stdsc.transform(np.array(df_test[var]).reshape(-1, 1))


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


def OHE_proccesing(df, var):
    # Drop ensures reduced correlation between the OneHot encoded categories
    embark_ohe = OneHotEncoder(categories="auto", drop="first")
    return embark_ohe.fit_transform(np.array(df[var]).reshape(-1, 1)).toarray().tolist()
