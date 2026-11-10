"""Candidate models. A simple baseline first, so the ANN has something to beat."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def get_model(name: str, random_state: int = 5):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if name == "mlp":
        # sklearn stand-in for your ANN: one hidden layer of 20 units, early stopping.
        return MLPClassifier(
            hidden_layer_sizes=(20,), early_stopping=True, max_iter=300, random_state=random_state
        )
    if name == "keras_ann":
        # TODO (portfolio task): port the notebook's Keras model into a scikit-learn compatible
        # wrapper (e.g. scikeras.KerasClassifier) with a validation split and class weights.
        raise NotImplementedError("Port the Keras ANN from the original notebook here.")
    raise ValueError(f"Unknown model: {name}")
