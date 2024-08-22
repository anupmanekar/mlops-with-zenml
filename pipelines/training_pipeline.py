import logging

from zenml import pipeline

from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.preprocess_data import clean_data
from steps.train_model import train_model


@pipeline
def train_pipeline(path: str):
    logging.info("Creating pipeline")
    data = ingest_data(path)
    x_train, x_test, y_train, y_test = clean_data(data)
    trained_model = train_model(x_train, x_test, y_train, y_test)
    r2, mse = evaluate_model(trained_model, x_test, y_test)
    logging.info(f"R2 Score: {r2}, MSE: {mse}")