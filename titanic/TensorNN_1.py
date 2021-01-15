import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

input_data = pd.read_csv("train.csv")
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


features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

test_data = feature_eng(input_data)

evidence = test_data[features].values.tolist()
labels = test_data["Survived"].values.tolist()

# Split the data randomly into training and test set (test size is 40%)
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.2
)

# Convert the labels into categorical (one-hot encoding)
y_training = tf.keras.utils.to_categorical(y_training)
y_testing = tf.keras.utils.to_categorical(y_testing)

# Convert the evidence list into a numpy array ()
X_training = np.array(X_training)
X_testing = np.array(X_testing)

# Create a neural network (sequential neural network in this case)
model = tf.keras.models.Sequential()

# Add a hidden layer with 24 units, with ReLU activation
model.add(tf.keras.layers.Dense(14, input_shape=(7,), activation="relu"))

# Add hidden layer with 1 unit, with relu activation
model.add(tf.keras.layers.Dense(28, activation="relu"))

# Add output layer with 2 units, with sigmoid activation function for the probability
model.add(tf.keras.layers.Dense(2, activation="sigmoid"))

# Train neural network (how to optimize, which loss function, which metric to evaluate how good the model is)
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
# Train the model (epochs=20 means, we will go through each data point 20 times?!)
model.fit(X_training, y_training, epochs=30)

# Evaluate how well model performs
model.evaluate(X_testing, y_testing, verbose=2)
