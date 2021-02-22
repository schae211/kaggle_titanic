# Import models from sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Import the ray library
from ray import tune

# Define BaseEstimators here?
svc_base = SVC(kernel="rbf", gamma=0.05, probability=False, C=23)
nusvc_base = NuSVC(kernel="rbf", gamma="auto", probability=False, nu=0.4)
knn_base = KNeighborsClassifier(n_neighbors=10, weights="uniform", algorithm="ball_tree")


dtc_config = {
    "class": DecisionTreeClassifier,
    "criterion": tune.choice(["gini", "entropy"]),
    "max_features": tune.choice(["auto", "sqrt", "log2"]),
    "random_state": 1
}


bc_config = {
    "class": BaggingClassifier,
    # "base_estimator": DecisionTreeClassifier(),
    # How to use something else than DecisionTreeClassifier?
    "base_estimator": tune.choice([SVC(kernel="rbf", gamma=0.05, probability=False, C=23), NuSVC(kernel="rbf", gamma="auto", probability=False, nu=0.4), KNeighborsClassifier(n_neighbors=10, weights="uniform", algorithm="ball_tree"), DecisionTreeClassifier()]),
    "n_estimators": tune.randint(5, 20),
    "max_samples": tune.randint(1, 3),
    "max_features": tune.randint(1, 3),
    "random_state": 1,
    "algorithm": "SAMME"
}

abc_config = {
    "class": AdaBoostClassifier,
    # "base_estimator": DecisionTreeClassifier(),
    # How to use something else than DecisionTreeClassifier?
    "base_estimator": tune.choice([SVC(kernel="rbf", gamma=0.05, probability=False, C=23), NuSVC(kernel="rbf", gamma="auto", probability=False, nu=0.4), KNeighborsClassifier(n_neighbors=10, weights="uniform", algorithm="ball_tree"), DecisionTreeClassifier()]),
    "n_estimators": tune.qrandint(10, 100, 10),
    "learning_rate": tune.uniform(0.5, 1.5),
    "random_state": 1,
    "algorithm": "SAMME"
}

knn_config = {
    "class": KNeighborsClassifier,
    "n_neighbors": tune.randint(3, 20),
    "weights": tune.choice(["uniform", "distance"]),
    "algorithm": tune.choice(["auto", "ball_tree", "kd_tree", "brute"]),
    "leaf_size": tune.qrandint(10, 60, 5),
}

rf_config = {
    "class": RandomForestClassifier,
    "max_depth": tune.randint(4, 10),
    "min_samples_leaf": tune.randint(1, 50),
    "n_estimators": tune.qrandint(lower=20, upper=250, q=10),
    "criterion": tune.choice(["gini", "entropy"]),
    "max_features": tune.choice(["auto", "sqrt", "log2"]),
    'random_state': 1
}

svm_config = {
    "class": SVC,
    "kernel": tune.choice(['rbf', 'sigmoid']),  # poly: too slow
    "gamma": tune.choice(['scale', 'auto', 0.01, 0.05, 0.1, 0.5, 1, 2]),
    "probability": tune.choice([True, False]),
    "C": tune.randint(1, 25),
    "random_state": 1
}

gbc_config = {
    "class": GradientBoostingClassifier,
    "n_estimators": tune.qrandint(lower=20, upper=250, q=10),
    "learning_rate": tune.choice([0.01, 0.05, 0.1, 0.2, 0.5, 1]),
    "criterion": tune.choice(["friedman_mse", "mse"]),
    "max_depth": tune.randint(2, 10),
    "loss": tune.choice(["deviance", "exponential"]),
    "random_state": 1
}

nusvc_config = {
    "class": NuSVC,
    "nu": tune.choice([0.3, 0.4, 0.5, 0.6, 0.7]),
    "kernel": tune.choice(['rbf', 'sigmoid']),  # poly: too slow
    "gamma": tune.choice(['scale', 'auto', 0.01, 0.05, 0.1, 0.5, 1, 2]),
    "probability": tune.choice([True, False]),
    "cache_size": tune.choice([500]),
    "random_state": 1
}