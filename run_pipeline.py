from pipelines.training_pipeline import train_pipeline
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.preprocess_data import clean_data
from steps.train_model import train_model

if __name__ == "__main__":
    train_pipeline("./data/olist_customers_dataset.csv", ingest_data(), clean_data(), train_model(), evaluate_model())