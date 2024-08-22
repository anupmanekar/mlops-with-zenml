import logging

from zenml import step
import pandas as pd

class IngestData:
    def __init__(self, path: str):
        self.path = path

    def run(self):
        logging.info(f"Reading data from {self.path}")
        return pd.read_csv(self.path)
    
@step(enable_cache=True)
def ingest_data(path: str) -> pd.DataFrame:
    return IngestData(path).run()
