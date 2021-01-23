from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_processing_2 import df_train, df_test


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

features = ["Sex", "Age_StdSc", "Fare_StdSc", "Name_StdSc", "SibSp_StdSc", "Parch_StdSc", "Pclass_StdSc", "Embarked_StdSc"]

evidence = prep_evidence(df_train, features)
labels = df_train["Survived"].values.tolist()

X_test = prep_evidence(df_test, features)

# Split the data randomly into training and test set (test size is 40%)
X_train, X_val, y_train, y_val = train_test_split(evidence, labels, test_size=0.4)

# Validate:
print(len(evidence[0]))
print(len(X_test[0]))

# Chose the model
# model = SVC(kernel="rbf", C=1, random_state=10, gamma=0.2)
model = RandomForestClassifier(criterion="gini", n_estimators=50)
# model = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")


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
output.to_csv('submission_RFC.csv', index=False)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(len(X_train[0])):
    print("%2d) %-*s %f" % (f + 1, 30,
                            features[indices[f]],
                            importances[indices[f]]))
