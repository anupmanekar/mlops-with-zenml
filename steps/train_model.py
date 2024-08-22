import logging
import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from model.training_models import RandomForestModel
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
    
@step(experiment_tracker=experiment_tracker)
def train_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> RegressorMixin:
    """
        Training model steps
        Args:
            x_train: pd.DataFrame : Training data
            x_test: pd.DataFrame : Testing data
            y_train: pd.DataFrame : Training labels
            y_test: pd.DataFrame : Testing labels
        Returns: Trained model
    """
    try:
        logging.info("Training model")
        mlflow.sklearn.autolog()
        # Initialising the model
        model = RandomForestModel()
        # Running the model
        return model.train(x_train, y_train)
    except Exception as e:
        logging.error(f"Error while training model: {e}")
        raise e