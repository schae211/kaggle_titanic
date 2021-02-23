# config file to easily to adjust hyper parameters


# Training parameters
training_parameter = {"epochs": 500,
                      "learning_rate": 0.0001,
                      "alpha": 0.25,
                      "gamma": 2.0,
                      # "features": ["Sex", "SibSp", "Age", "Pclass", "Parch", "Fare"],
                      "features": ["Sex", "SibSp", "Age", "Pclass", "Parch"],
                      "batch_size": 16,
                      "patience": 10}

network_parameter = {"hidden_layer": 1,
                     "hidden_neurons": 64,
                     "l1": 0.001,
                     "l2": 0.001,
                     "threshold": 0.5}


cfg = {'training': training_parameter, 'network': network_parameter}

# Optimized ML models
# This dictionary should contain the optimized hyperparameters for all ML models according to RayTune
ml_cfg = {
    "knn": {
        "n_neighbors": 17,
        "weights": "uniform",
        "algorithm": "ball_tree",
        "leaf_size": 50
    },
    "svc": {
        "kernel": "rbf",
        "gamma": 0.1,
        "probability": False,
        "C": 12
    },
    "nusvc": {
        "nu": 0.4,
        "kernel": "rbf",
        "gamma": "auto",
        "probability": True
    },
    "rf": {
        "max_depth": 10,
        "min_samples_leaf": 2,
        "n_estimators": 70,
        "criterion": "gini",
        "max_features": "log2"
    },
    "gbc": {
        "n_estimators": 100,
        "learning_rate": 0.5,
        "criterion": "mse",
        "max_depth": 3,
        "loss": "exponential"
},
    "dtc": {
        "criterion": "entropy",
        "max_features": "auto"
    }
}