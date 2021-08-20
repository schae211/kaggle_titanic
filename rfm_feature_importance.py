"""
The Random Forest Model can be used to access the importance of certain features
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from feature_processing_3 import load_and_process_data_set
from cfg import ml_cfg

df_train, df_test = load_and_process_data_set()

def flatten(df, feat):
    """
    :param df: dataframe
    :param feat: features of interest
    :return: flat (not nested) list of the features from the dataframe
    """
    primary_list = df[feat].values.tolist()
    evidence_list = []
    for ev in primary_list:
        temp = []
        for element in ev:
            if type(element) != list:
                temp.append(element)
            else:
                for number in element:
                    temp.append(number)
        evidence_list.append(temp)
    return evidence_list

features = ['Sex', 'Age_StdSc', 'Pclass_Scale', 'Fare_StdSc', 'SibSp', 'Parch']

X_train = np.array(flatten(df_train, features))
y_train = df_train["Survived"].values.tolist()

rfm = RandomForestClassifier(n_estimators=ml_cfg["rf"]["n_estimators"], max_depth=ml_cfg["rf"]["max_depth"],
                            min_samples_leaf=ml_cfg["rf"]["min_samples_leaf"], criterion=ml_cfg["rf"]["criterion"],
                            max_features=ml_cfg["rf"]["max_features"])
rfm.fit(X_train, y_train)
importances = rfm.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(X_train[0])):
    print("%2d) %-*s %f" % (f + 1, 30,
                            features[indices[f]],
                            importances[indices[f]]))
