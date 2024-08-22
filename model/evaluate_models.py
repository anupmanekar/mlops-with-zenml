import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluation(ABC):
    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class MeanSquaredError(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        logging.info("Calculating Mean Squared")
        return mean_squared_error(y_true, y_pred)

class MeanAbsoluteError(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        logging.info("Calculating Mean Absolute")
        return mean_absolute_error(y_true, y_pred)

class R2Score(Evaluation):
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        logging.info("Calculating R2 Score")
        return r2_score(y_true, y_pred)