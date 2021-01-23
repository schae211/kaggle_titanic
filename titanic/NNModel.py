import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Softmax, BatchNormalization, Dropout, Concatenate
from tensorflow.keras import Model


from loss_function import custom_loss_function, binary_focal_loss_fixed
from feature_processing_1 import load_and_process_data_set
from cfg import cfg


def flatten_nested_list(nested_list):
    un_nested_list = []
    for entry in nested_list:
        flatten_entry = []
        for val in entry:
            if isinstance(val, list):
                flatten_entry.extend(val)
            else:
                flatten_entry.append(val)
        un_nested_list.append(flatten_entry)
    return un_nested_list


def load_and_create_data_partition():
    train_data, test_data = load_and_process_data_set()

    features = cfg['training']['features']

    training_data = train_data.copy()
    test_data = test_data.copy()

    evidence = training_data[features].values.tolist()
    labels = training_data["Survived"].values.tolist()
    test_data = test_data[features].values.tolist()

    # Split the data randomly into training and test set (test size is 40%)
    X_train, X_val, y_train, y_val = train_test_split(evidence, labels, random_state=200)

    # Convert the labels into categorical (what happens here under the hood?)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)

    # Convert the evidence list into a numpy array (why is that necessary?)
    X_train = np.array(flatten_nested_list(X_train))
    X_val = np.array(flatten_nested_list(X_val))
    X_test = np.array(flatten_nested_list(test_data))
    return X_train, X_val, X_test, y_train, y_val


def create_model():
    # Create a neural network (sequential neural network in this case)

    inputs = Input(shape=(6,))
    x = Dense(128, input_shape=(len(cfg['training']['features']),), activation="relu",
              kernel_regularizer=tf.keras.regularizers.l1(cfg['network']['l1']),
              activity_regularizer=tf.keras.regularizers.l2(cfg['network']['l2']),
              use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(32, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l1(cfg['network']['l1']),
              activity_regularizer=tf.keras.regularizers.l2(cfg['network']['l2']),
              use_bias=False)(x)
    x = BatchNormalization()(x)
    x_skip = x

    for _ in range(cfg['network']['hidden_layer']):
        # Add hidden layer with 1 unit, with relu activation
        x = Dense(cfg['network']['hidden_neurons'], activation="relu",
                  kernel_regularizer=tf.keras.regularizers.l1(cfg['network']['l1']),
                  activity_regularizer=tf.keras.regularizers.l2(cfg['network']['l2']), use_bias=False)(x)
        x = BatchNormalization()(x)

    x = x + x_skip

    # x_sigma = Dense(1, activation="softplus")(x)
    # Add output layer with 2 units, with sigmoid activation function for the probability
    x_class = Dense(2, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l1(cfg['network']['l1']),
                    activity_regularizer=tf.keras.regularizers.l2(cfg['network']['l2']))(x)
    # x_class = Softmax()(x_class)
    # outputs = Concatenate()([x_class, x_sigma])

    nn_model = Model(inputs, x_class)

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
            if pred[j, 0] < pred[j, 1]:
                confusion_mat['TP'] += 1
            else:
                confusion_mat['FN'] += 1
        else:
            if pred[j, 0] < pred[j, 1]:
                confusion_mat['FP'] += 1
            else:
                confusion_mat['TN'] += 1
    print(confusion_mat)
    conf_mat = np.array([[confusion_mat["TP"], confusion_mat["FP"]],
                         [confusion_mat["FN"], confusion_mat["TN"]]])
    df_cm = pd.DataFrame(conf_mat, index=[c for c in ["survived", "dead"]],
                         columns=[j for j in ["survived", "dead"]])
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
    submission_data = pd.read_csv("Data/test.csv")
    X_training, X_validation, X_test, y_training, y_validation = load_and_create_data_partition()

    # callbacks
    earlyStopping = EarlyStopping(monitor='val_loss', patience=cfg['training']['patience'], verbose=1, mode='min')
    mcp_save = ModelCheckpoint('./Models/best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model (epochs=20 means, we will go through each data point 20 times?!)
    model = create_model()
    training_history = model.fit(X_training, y_training, epochs=cfg['training']['epochs'],
                                 validation_data=(X_validation, y_validation), batch_size=cfg['training']['batch_size'],
                                 callbacks=[mcp_save, earlyStopping], verbose=2)
    plot_training_history(training_history)

    # custom objects to load the custom loss function is needed
    model = tf.keras.models.load_model("./Models/best_model.h5",
                                       custom_objects={'binary_focal_loss_fixed': binary_focal_loss_fixed})
    # evaluate the performance
    predictions_validation = model.predict(X_validation)
    predictions_val_class = predictions_validation[:, :2]
    # predictions_val_sigma = predictions_validation[:, 2]
    evaluate_model(predictions_val_class, y_validation)

    predictions = model.predict(X_test)[:, :2]
    results = np.zeros(len(predictions), dtype=int)

    for i in range(len(predictions)):
        if predictions[i, 0] > predictions[i, 1]:
            results[i] = 0
        else:
            results[i] = 1

    # Safe predictions:
    output = pd.DataFrame({'PassengerId': submission_data.PassengerId, 'Survived': results})
    output.to_csv('./submissions/submission_tensorNN_2.csv', index=False)
