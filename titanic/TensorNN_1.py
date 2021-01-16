import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

training_data = pd.read_csv("train.csv")
submission_data = pd.read_csv("test.csv")


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss
    return binary_focal_loss_fixed


def feature_eng(df):
    df["Sex"] = np.where(df["Sex"] == "female", 0, 1)

    df = df.fillna(df.median())

    for index in df.index:
        if df.loc[index, "Embarked"] == "S":
            df.loc[index, "Embarked"] = 0
        elif df.loc[index, "Embarked"] == "C":
            df.loc[index, "Embarked"] = 1
        else:
            df.loc[index, "Embarked"] = 2
    return df


# features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
features = ["Sex", "SibSp", "Parch", "Age"]

training_data = feature_eng(training_data)
test_data = feature_eng(submission_data)

evidence = training_data[features].values.tolist()
labels = training_data["Survived"].values.tolist()

test_data = test_data[features].values.tolist()

# Split the data randomly into training and test set (test size is 40%)
X_training, X_validation, y_training, y_validation = train_test_split(evidence, labels, test_size=0.2)

# Convert the labels into categorical (what happens here under the hood?)
y_training = tf.keras.utils.to_categorical(y_training)
y_validation = tf.keras.utils.to_categorical(y_validation)

# Convert the evidence list into a numpy array (why is that necessary?)
X_training = np.array(X_training)
X_validation = np.array(X_validation)
X_test = np.array(test_data)

# Create a neural network (sequential neural network in this case)
model = tf.keras.models.Sequential()

# Add a hidden layer with 24 units, with ReLU activation
model.add(tf.keras.layers.Dense(32, input_shape=(len(features),), activation="relu"))

# Add hidden layer with 1 unit, with relu activation
model.add(tf.keras.layers.Dense(64, activation="relu"))

# Add output layer with 2 units, with sigmoid activation function for the probability
model.add(tf.keras.layers.Dense(2, activation="sigmoid"))

# Train neural network (how to optimize, which loss function, which metric to evaluate how good the model is)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, name='Adam')

model.compile(
    optimizer=optimizer,
    # optimizer="adam",
    # loss="binary_crossentropy",
    loss=binary_focal_loss(),
    metrics=["accuracy"]
)
# Train the model (epochs=20 means, we will go through each data point 20 times?!)
training_history = model.fit(X_training, y_training, epochs=100,
                             validation_data=(X_validation, y_validation), batch_size=16)

# Evaluate how well model performs
model.evaluate(X_validation, y_validation, verbose=2)
plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predictions = model.predict(X_test)
results = np.zeros(len(predictions), dtype=int)

for i in range(len(predictions)):
    if predictions[i, 0] > predictions[i, 1]:
        results[i] = 0
    else:
        results[i] = 1

# Safe predictions:
output = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': results})
output.to_csv('submission_tensorNN_2.csv', index=False)
