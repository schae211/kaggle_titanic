# config file to easily to adjust hyper parameters


# Training parameters
training_parameter = {"epochs": 200,
                      "learning_rate": 0.0001,
                      "alpha": 0.25,
                      "gamma": 2.0,
                      "features": ["Sex", "SibSp", "Age", "Pclass", "Parch", "Fare"]}


cfg = {'training': training_parameter, 'network': dict()}
