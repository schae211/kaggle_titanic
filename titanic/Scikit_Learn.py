# Import models from sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Import the ray library
from ray import tune

# Define BaseEstimators here?
svc_base = SVC(kernel="rbf", gamma=0.05, probability=False, C=23)
nusvc_base = NuSVC(kernel="rbf", gamma="auto", probability=False, nu=0.4)
knn_base = KNeighborsClassifier(n_neighbors=10, weights="uniform", algorithm="ball_tree")


from feature_processing_3 import load_and_process_data_set

df_train, df_test = load_and_process_data_set()


def prep_evidence(df, feat):
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

# df_train["Embarked_int"] = df_train["Embarked"].apply(lambda row: 1 if row == "S" else 0)
# df_test["Embarked_int"] = df_test["Embarked"].apply(lambda row: 1 if row == "S" else 0)
features = ["Sex", "Age_StdSc", "Pclass_Scale", "Fare_StdSc", "Name_StdSc", "SibSp"]
# features = ["Sex", "Age_StdSc", "Fare_StdSc", "Pclass", "SibSp", "Parch"]

evidence = prep_evidence(df_train, features)
labels = df_train["Survived"].values.tolist()

X_test = prep_evidence(df_test, features)

# Split the data randomly into training and test set (test size is 40%)
#X_train, X_val, y_train, y_val = train_test_split(evidence, labels, test_size=0.2, random_state=100, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(evidence, labels, test_size=0.05, stratify=labels)

# Validate:
print(len(evidence[0]))
print(len(X_test[0]))

# Chose the model
# model = SVC(kernel="rbf", C=1, random_state=100, gamma=0.2)
# model = RandomForestClassifier(criterion="gini", n_estimators=50)
# model = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
model = AdaBoostClassifier(n_estimators=100, base_estimator=nusvc_base, learning_rate=1, algorithm='SAMME')


# Fit model
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_val)

# Compute how well we performed
correct = (y_val == predictions).sum()
incorrect = (y_val != predictions).sum()
total = len(predictions)

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")

test_predictions = model.predict(X_test)

# Safe predictions:
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': test_predictions})
output.to_csv('AdaBoost_NuSVC_2.csv', index=False)