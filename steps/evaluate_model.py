import logging
import mlflow
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from sklearn.base import RegressorMixin

from zenml import step

from model.evaluate_models import MeanSquaredError, R2Score, MeanAbsoluteError

@step
def evaluate_model(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
    )   -> Tuple[Annotated[float, 'r2_score'], Annotated[float, 'mse_score']]:
        prediction = model.predict(x_test)
        mse_class = MeanSquaredError()
        mse = mse_class.evaluate(y_test, prediction)
        mlflow.log_metric("mse", mse)
        
        r2_class = R2Score()
        r2 = r2_class.evaluate(y_test, prediction)
        mlflow.log_metric("r2", r2)

        mae_class = MeanAbsoluteError()
        mae = mae_class.evaluate(y_test, prediction)
        mlflow.log_metric("mae", mae)
        logging.info(f"R2 Score: {r2}, MSE: {mse}, MAE: {mae}")
        return r2, mse
