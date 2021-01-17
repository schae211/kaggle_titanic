import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


test_data = pd.read_csv("Data/train.csv")
submission_data = pd.read_csv("Data/test.csv")


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

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

test_data = feature_eng(test_data)

evidence = test_data[features].values.tolist()
labels = test_data["Survived"].values.tolist()

# Split the data randomly into training and test set (test size is 40%)
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=3)
model = GaussianNB()

# Fit model
model.fit(X_training, y_training)

# Make predictions on the testing set
predictions = model.predict(X_testing)

# Compute how well we performed
correct = (y_testing == predictions).sum()
incorrect = (y_testing != predictions).sum()
total = len(predictions)

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")

