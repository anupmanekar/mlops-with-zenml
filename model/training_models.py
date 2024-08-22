import logging
from lightgbm import LGBMRegressor
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        """
            Train the model
            Args:   x_train: pd.DataFrame : Training data
                    y_train: pd.DataFrame : Training labels
            Returns: Trained model
        """
        pass
    
    def test(self, trial, x_train, x_test, y_train, y_test):
        """
            Test the model
            Args:   trial: Optuna.Trial : Optuna trial object
                    x_train: pd.DataFrame : Training data
                    x_test: pd.DataFrame : Testing data
                    y_train: pd.DataFrame : Training labels
                    y_test: pd.DataFrame : Testing labels
            Returns: Optuna trial object
        """
        pass

class RandomForestModel(Model):
    def train(self, x_train, y_train, **kwargs):
        logging.info("Training Random Forest Model")
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg
    
    def test(self, trial, x_train, x_test, y_train, y_test):
        logging.info("Optimizing Random Forest Model")
        n_estimators=trial.suggest_int("n_estimators", 1, 200)
        max_depth=trial.suggest_int("max_depth", 1, 20)
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20)
        reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        return reg.score(x_test, y_test)

class LightGBMModel(Model):
    def train(self, x_train, y_train, **kwargs):
        logging.info("Training LightGBM Model")
        reg = LGBMRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg

class XGBoostModel(Model):
    def train(self, x_train, y_train, **kwargs):
        logging.info("Training XGBoost Model")
        reg = XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        return reg