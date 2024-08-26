import logging

from zenml import pipeline


@pipeline
def train_pipeline(path: str, ingest_data, clean_data, train_model, evaluate_model):
    logging.info("Creating pipeline")
    data = ingest_data(path)
    x_train, x_test, y_train, y_test = clean_data(data)
    trained_model = train_model(x_train, x_test, y_train, y_test)
    r2, mse = evaluate_model(trained_model, x_test, y_test)
    logging.info(f"R2 Score: {r2}, MSE: {mse}")