import textwrap

import numpy as np
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelConfig:
    """
    Data class to encapsulate a machine learning model configuration.

    Attributes:
    - name: Name of the estimator
    - model: Machine learning model instance.
    - hyperparameters: Dictionary of hyperparameters and their possible values.
    """
    name: str
    model: object
    hyperparameters: dict

    def __str__(self):
        wrapper = textwrap.TextWrapper(width=100) 

        s = 'ModelConfig(\n'
        s += f'\tModel name: {self.name}\n'
        s += '\tModel hyperparameters{\n'
        for key, value in self.hyperparameters.items():
            s += f"\t\t{key}: {value}\n"
        s += '\t}\n'
        s += ')'
        return s

    def __repr__(self):
        return f"ModelConfig(name='{self.name}', model={repr(self.model)}, hyperparameters={repr(self.hyperparameters)})"


def create_logistic_regression_config():
    """
    Creates a ModelConfig instance for Logistic Regression.

    Returns:
    ModelConfig: Configuration for Logistic Regression model.
    """
    return ModelConfig(
        name='Logistic Regression',
        model=LogisticRegression(),
        hyperparameters={
            'logisticregression__penalty': ('l1', 'l2', 'elasticnet', None),
            'logisticregression__C': np.concatenate((np.arange(0, 1, 0.1), np.arange(1, 10, 1), np.arange(10, 101, 10))),
            'logisticregression__fit_intercept': (False, True),
            'logisticregression__intercept_scaling': [0, 1, 2],
            'logisticregression__class_weight': ('balanced', None, {False: 1, True: 1.5}),
            'logisticregression__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'logisticregression__multi_class': ('auto', 'ovr', 'multinomial'),
            'logisticregression__l1_ratio': np.arange(0, 1.1, 0.1)
        }
    )


def create_decision_tree_config():
    """
    Creates a ModelConfig instance for Decision Tree.

    Returns:
    ModelConfig: Configuration for Decision Tree model.
    """
    return ModelConfig(
        name='Decision Tree',
        model=DecisionTreeClassifier(),
        hyperparameters={
            'decisiontreeclassifier__criterion': ('gini', 'entropy', 'log_loss'),
            'decisiontreeclassifier__splitter': ('best', 'random'),
            'decisiontreeclassifier__max_features': (None, 1, 4, 9, 'auto', 'sqrt', 'log2'),
            'decisiontreeclassifier__max_leaf_nodes': (1, 5, 10, 20, None),
            'decisiontreeclassifier__class_weight': ('balanced', None, {False: 1, True: 1.5}),
        }
    )


def create_random_forest_config():
    """
    Creates a ModelConfig instance for Random Forest.

    Returns:
    ModelConfig: Configuration for Random Forest model.
    """
    return ModelConfig(
        name='Random Forest',
        model=RandomForestClassifier(),
        hyperparameters={
            'randomforestclassifier__n_estimators': np.concatenate((np.array([50, 300]), np.arange(500, 4501, 1000))),
            'randomforestclassifier__criterion': ('gini', 'entropy', 'log_loss'),
            'randomforestclassifier__max_depth': (3, 6, 13, 21),
            'randomforestclassifier__min_samples_split': (2, 4, 8, 12),
            'randomforestclassifier__min_samples_leaf': (3, 7, 9),
            'randomforestclassifier__max_features': ('sqrt', 'log2', None, 2, 7),
            'randomforestclassifier__oob_score': (False, True),
            'randomforestclassifier__class_weight': ('balanced', None),
        }
    )



"""Example:

from model_config import create_logistic_regression_config, create_decision_tree_config, create_random_forest_config

# Create instances of ModelConfig for each model
logreg_config = create_logistic_regression_config()
decision_tree_config = create_decision_tree_config()
random_forest_config = create_random_forest_config()

# Example usage of the configurations
for model_config in [logreg_config, decision_tree_config, random_forest_config]:
    model_instance = model_config.model
    hyperparameters = model_config.hyperparameters
    print(f"Model: {model_instance}")
    print("Hyperparameters:")
    for key, values in hyperparameters.items():
        print(f"  {key}: {values}")


"""