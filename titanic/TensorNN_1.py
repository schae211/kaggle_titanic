import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from loss_function import binary_focal_loss_fixed
from cfg import cfg


def feature_eng(df):
    df["Sex"] = np.where(df["Sex"] == "female", 0, 1)

    # df = df.fillna(df.median())
    df = df.fillna(0)

    # Normalize values between 0 and 1
    df["Age"] /= df["Age"].max()
    df["Fare"] /= df["Fare"].max()
    df["Parch"] /= df["Parch"].max()
    df["Pclass"] /= df["Pclass"].max()
    df["SibSp"] /= df["SibSp"].max()

    for index in df.index:
        if df.loc[index, "Embarked"] == "S":
            df.loc[index, "Embarked"] = 0
        elif df.loc[index, "Embarked"] == "C":
            df.loc[index, "Embarked"] = 0.5
        else:
            df.loc[index, "Embarked"] = 1
    return df


def load_and_create_data_partition(train_data, test_data):
    features = cfg['training']['features']

    training_data = feature_eng(train_data)
    test_data = feature_eng(test_data)

    evidence = training_data[features].values.tolist()
    labels = training_data["Survived"].values.tolist()

    test_data = test_data[features].values.tolist()

    # Split the data randomly into training and test set (test size is 40%)
    X_train, X_val, y_train, y_val = train_test_split(evidence, labels, test_size=0.2, random_state=200)

    # Convert the labels into categorical (what happens here under the hood?)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)

    # Convert the evidence list into a numpy array (why is that necessary?)
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(test_data)
    return X_train, X_val, X_test, y_train, y_val


def create_model():
    # Create a neural network (sequential neural network in this case)
    nn_model = tf.keras.models.Sequential()

    # Add a hidden layer with 24 units, with ReLU activation
    nn_model.add(tf.keras.layers.Dense(128, input_shape=(len(cfg['training']['features']),), activation="relu",
                                       kernel_regularizer=tf.keras.regularizers.l1(cfg['network']['l1']),
                                       activity_regularizer=tf.keras.regularizers.l2(cfg['network']['l2']),
                                       use_bias=False))
    nn_model.add(tf.keras.layers.BatchNormalization())
    nn_model.add(tf.keras.layers.Dropout(0.4))

    for _ in range(cfg['network']['hidden_layer']):
        # Add hidden layer with 1 unit, with relu activation
        nn_model.add(tf.keras.layers.Dense(cfg['network']['hidden_neurons'], activation="relu",
                                           kernel_regularizer=tf.keras.regularizers.l1(cfg['network']['l1']),
                                           activity_regularizer=tf.keras.regularizers.l2(cfg['network']['l2']),
                                           use_bias=False))
        nn_model.add(tf.keras.layers.BatchNormalization())
        nn_model.add(tf.keras.layers.Dropout(0.4))

    # Add output layer with 2 units, with sigmoid activation function for the probability
    nn_model.add(tf.keras.layers.Dense(2, activation="sigmoid",
                                       kernel_regularizer=tf.keras.regularizers.l1(cfg['network']['l1']),
                                       activity_regularizer=tf.keras.regularizers.l2(cfg['network']['l2'])))
    nn_model.add(tf.keras.layers.Softmax())

    # Train neural network (how to optimize, which loss function, which metric to evaluate how good the model is)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['training']['learning_rate'], name='Adam')

    nn_model.compile(
        optimizer=optimizer,
        # optimizer="adam",
        # loss="binary_crossentropy",
        loss=binary_focal_loss_fixed,
        metrics=["accuracy"])
    nn_model.summary()
    return nn_model


def plot_training_history(history):
    # Evaluate how well model performs
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def evaluate_model(pred, label):
    # calculate TP, FP, TN, FN
    # 0 = Dead
    # 1 = Survived (positive class)
    confusion_mat = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for j in range(len(pred)):
        if label[j, 1] == 1:  # positive class
            if pred[j, 0] <= cfg['network']['threshold']:
                confusion_mat['TP'] += 1
            else:
                confusion_mat['FN'] += 1
        else:
            if pred[j, 0] <= cfg['network']['threshold']:
                confusion_mat['FP'] += 1
            else:
                confusion_mat['TN'] += 1
    print(confusion_mat)
    conf_mat = np.array([[confusion_mat["TP"], confusion_mat["FP"]],
                         [confusion_mat["FN"], confusion_mat["TN"]]])
    df_cm = pd.DataFrame(conf_mat, index=[i for i in ["survived", "dead"]],
                         columns=[i for i in ["survived", "dead"]])
    plt.xlabel("True class")
    plt.ylabel("Predicted class")
    sn.heatmap(df_cm, annot=True)
    plt.show()

    tp = confusion_mat["TP"]
    fp = confusion_mat["FP"]
    tn = confusion_mat["TN"]
    fn = confusion_mat["FN"]

    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2 * precision * recall) / (precision + recall)

    print("Accuracy: ", round(accuracy, 2))
    print("Precision: ", round(precision, 2))
    print("Recall: ", round(recall, 2))
    print("F1 Score: ", round(f1_score, 2))


if __name__ == "__main__":
    training_data = pd.read_csv("Data/train.csv")
    submission_data = pd.read_csv("Data/test.csv")
    X_training, X_validation, X_test, y_training, y_validation = load_and_create_data_partition(training_data,
                                                                                                submission_data)

    # callbacks
    earlyStopping = EarlyStopping(monitor='val_loss', patience=cfg['training']['patience'], verbose=1, mode='min')
    mcp_save = ModelCheckpoint('./Models/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model (epochs=20 means, we will go through each data point 20 times?!)
    model = create_model()
    training_history = model.fit(X_training, y_training, epochs=cfg['training']['epochs'],
                                 validation_data=(X_validation, y_validation), batch_size=cfg['training']['batch_size'],
                                 callbacks=[earlyStopping, mcp_save])
    plot_training_history(training_history)

    # custom objects to load the custom loss function is needed
    model = tf.keras.models.load_model("./Models/best_model.h5",
                                       custom_objects={'binary_focal_loss_fixed': binary_focal_loss_fixed})
    # evaluate the performance
    predictions = model.predict(X_validation)
    evaluate_model(predictions, y_validation)

    predictions = model.predict(X_test)
    results = np.zeros(len(predictions), dtype=int)

    for i in range(len(predictions)):
        if predictions[i, 0] > cfg['network']['threshold']:
            results[i] = 0
        else:
            results[i] = 1

    # Safe predictions:
    output = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': results})
    output.to_csv('./submissions/submission_tensorNN_2.csv', index=False)
