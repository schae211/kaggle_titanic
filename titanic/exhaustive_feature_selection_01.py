from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from feature_processing_3 import load_and_process_data_set
import numpy as np
import pandas as pd
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

features = ["Sex", "Age_StdSc", "Pclass_Scale", "Fare_StdSc", "Name_StdSc", "SibSp", "Parch", "Embarked"]
feature_names = ("Sex", "Age_StdSc", "Pclass_Scale", "Fare_StdSc", "Name_StdSc", "SibSp", "Parch", "Embarked")

X_train = np.array(flatten(df_train, features))
y_train = df_train["Survived"].values.tolist()

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

classifiers = [knn, svc, nusvc, rf, gbc]
classifiers_test = [knn, svc]

# For each specified model, run exhaustive sequential feature selection
for model in classifiers:
    efs = ExhaustiveFeatureSelector(model,
                                    min_features=3,
                                    max_features=len(features),
                                    scoring='accuracy',
                                    print_progress=True,
                                    # Specify to use all available CPUs -> "-1"
                                    n_jobs=-1,
                                    # Split for cross-validation
                                    cv=5)
    efs.fit(X_train, y_train, custom_feature_names=feature_names)
    df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
    df.sort_values('avg_score', inplace=True, ascending=False)
    name = f"{model}".split("(")[0]
    df.to_csv(f"Feature_Selection/{name}.csv")
    print(f"\nModel: {name}")
    print(f"Best accuracy score: {efs.best_score_}")
    print(f"Best subset (indices): {efs.best_idx_}")
    print(f"Best subset (corresponding names): {efs.best_feature_names_}")

