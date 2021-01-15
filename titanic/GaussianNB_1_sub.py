import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

test_data = pd.read_csv("train.csv")
submission_data = pd.read_csv("test.csv")


def feature_eng(df):
    df["Sex"] = np.where(df["Sex"] == "female", 0, 1)

    # fill NaN with the median/mean of the column
    df = df.fillna(df.median())

    for index in df.index:
        if df.loc[index, "Embarked"] == "S":
            df.loc[index, "Embarked"] = 0
        elif df.loc[index, "Embarked"] == "C":
            df.loc[index, "Embarked"] = 1
        else:
            df.loc[index, "Embarked"] = 2

    return df


test_data = feature_eng(test_data)
submission_data = feature_eng(submission_data)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

evidence = test_data[features].values.tolist()
labels = test_data["Survived"].values.tolist()

# Split the data randomly into training and test set (test size is 40%)
# X_training, X_testing, y_training, y_testing = train_test_split(evidence, labels, test_size=0.4)

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=3)
model = GaussianNB()

# Fit model
model.fit(evidence, labels)

# Make predictions on the testing set
X_sub = submission_data[features].values.tolist()
predictions = model.predict(X_sub)

# Safe predictions:
output = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_GaussianNB_1.csv', index=False)