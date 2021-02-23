from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from feature_processing_3 import load_and_process_data_set
import numpy as np
import pandas as pd
from datetime import datetime
from cfg import ml_cfg
import os

"""
Python Script dedicated to make the predictions for Kaggle using ML models from scikit (with timestamp, etc.)
"""


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
X_test = np.array(flatten(df_test, features))
y_train = df_train["Survived"].values.tolist()

# Specify the models
dtc = DecisionTreeClassifier(criterion=ml_cfg["dtc"]["criterion"], max_features=ml_cfg["dtc"]["max_features"])

knn = KNeighborsClassifier(n_neighbors=ml_cfg["knn"]["n_neighbors"], weights=ml_cfg["knn"]["weights"],
                           algorithm=ml_cfg["knn"]["algorithm"], leaf_size=ml_cfg["knn"]["leaf_size"])

svc = SVC(kernel=ml_cfg["svc"]["kernel"], gamma=ml_cfg["svc"]["gamma"], probability=ml_cfg["svc"]["probability"],
          C=ml_cfg["svc"]["C"])

nusvc = NuSVC(nu=ml_cfg["nusvc"]["nu"], kernel=ml_cfg["nusvc"]["kernel"], gamma=ml_cfg["nusvc"]["gamma"],
              probability=ml_cfg["nusvc"]["probability"])

rf = RandomForestClassifier(n_estimators=ml_cfg["rf"]["n_estimators"], max_depth=ml_cfg["rf"]["max_depth"],
                            min_samples_leaf=ml_cfg["rf"]["min_samples_leaf"], criterion=ml_cfg["rf"]["criterion"],
                            max_features=ml_cfg["rf"]["max_features"])

gbc = GradientBoostingClassifier(n_estimators=ml_cfg["gbc"]["n_estimators"], learning_rate=ml_cfg["gbc"]["learning_rate"],
                                 criterion=ml_cfg["gbc"]["criterion"], max_depth=ml_cfg["gbc"]["max_depth"],
                                 loss=ml_cfg["gbc"]["loss"])

bc = BaggingClassifier(base_estimator=svc)

abc = AdaBoostClassifier(base_estimator=dtc, algorithm="SAMME")

eclf = VotingClassifier(estimators=[('knn', knn), ('svc', svc), ('nusvc', nusvc), ("rf", rf), ("gbc", gbc), ("bc", bc)],
                        voting='hard')

model = eclf
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Safe predictions:
name = f"{model}".split("(")[0]
now = datetime.now()
now_string = now.strftime("(%d-%m-%Y_%H:%M:%S)")
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
directory = os.path.dirname(os.path.abspath("submissions"))
path = os.path.join(directory, f"submissions/{name}_{now_string}.csv")
output.to_csv(path, index=False)

