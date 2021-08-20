import numpy as np
import pandas as pd
from ray import tune
from copy import deepcopy
# from hyperopt import hp
from typing import Any, Callable, Dict
from ray.tune.suggest.optuna import OptunaSearch
from sklearn.model_selection import cross_val_score

"""
This python script uses the ray library to automate hyperparameter optimization for 
ML algorithms that are implemented in sci-kit learn
"""

# Import and load the dataframes
from feature_processing_3 import load_and_process_data_set
df_train, df_test = load_and_process_data_set()


# Define function that turns a nested list into a "flat list"
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

# Code Explanations for Python newbies like me
# "method: str" specifies that method must be of type string
# "-> tune.ExperimentAnalysis" specifies that return value must be of type tune.ExperimentAnalysis
def run_tune(method: str, num_samples: int) -> tune.ExperimentAnalysis:
    optuna_search = OptunaSearch(metric="mean_accuracy", mode="max")
    return tune.run(
        trial,
        config=methods[method],
        num_samples=num_samples,
        search_alg=optuna_search,
        # resources_per_trial={"gpu": 1, "cpu": 16}, # Using the GPU makes the search process a lot slower on my machine
        verbose=1,
        metric="mean_accuracy",   # otherwise I cannot get the dataframe in the end for some reason
        mode="max"                # See above
    )

# Define the "trainable function" that takes in the config dictionary (needs to have string keys and any value types)
def trial(config: Dict[str, Any]):
    _config = deepcopy(config)
    # e.g. model_class = RandomForestClassifier or SVC (see dictionary defintions above)
    model_class = config["class"]
    # e.g. pop the "class" key from the _config dictionary because it is no parameter for the model
    _config.pop("class", None)
    # Just like the normal scikit syntax we define our model using the config values
    model = model_class(
        **_config
    )
    # Get the cross validation score function from sci-kit learn
    # Training set is split into sets, model is trained with k-1 sets and unused set is used for validation
    # This is done 5 consecutive times (cv=5) and the performance is scored using the accuracy
    # Cross_Validation_Score, takes in a model that is fit to k-1 sets and returns the accuracy for the resulting set
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    # This basically sends the score to Tune
    tune.report(mean_accuracy=np.mean(scores), done=True)

# Wrapper function
def run_experiments(methods: list, num_samples: int = 50, num_results: int = 20):
    for method in methods:
        exp_analysis = run_tune(method, num_samples)
        df_results = exp_analysis.results_df
        # Get num_results number of best results
        df_results = df_results.sort_values("mean_accuracy", ascending=False).iloc[0:num_results, :]
        # Select the trial, accuracy column, Select the config column, concatenate and export to csv
        accuracy = df_results.iloc[0:num_results, 0:1]
        configs = df_results.iloc[0:num_results, 17:]
        pd.concat([accuracy, configs], axis=1).to_csv(f"RayTune/{method}.csv")
    return 1


# Define which features to use
features = ["Sex", "Age_StdSc", "Pclass_Scale", "Fare_StdSc", "Name_StdSc", "SibSp", "Parch", "Embarked"]

# Create list of features to train the models and a list of labels ("ground truth") to evaluate the models
X_train = flatten(df_train, features)
y_train = df_train["Survived"].values.tolist()

# Define Search Space for Ray Tune for each model (see sci-kit documentation of each model)
from ray_hyper_conf_1 import dtc_config, bc_config, abc_config, knn_config, rf_config, svm_config, gbc_config, nusvc_config

# List of methods: Dict[str, [Callable, Dict[str, Any]]]
methods = {"dtc": dtc_config, "bc": bc_config, "abc": abc_config, "knn": knn_config, "rf": rf_config,
           "svm": svm_config, "gbc": gbc_config, "nusvc": nusvc_config}

# Need to get bc and abc to work with other base estimators
methods_list = ["dtc", "bc", "abc", "knn", "rf", "svm", "gbc", "nusvc"]

methods_list_truncated = ["dtc", "knn", "rf", "svm", "gbc", "nusvc"]
# I still haven not figured out how to get those ensemble methods (bagging, boosting) to work with other base
# estimators here in RayTune, in ml_testing, it works perfectly
ensemble_methods = ["bc", "abc"]

run_experiments(methods_list_truncated, 250, 20)
