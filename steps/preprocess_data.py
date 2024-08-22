import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from model.data_preprocess import DataPreprocessStrategy, DataDivideStrategy, DataCleaning

    
@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'x_train'],
    Annotated[pd.DataFrame, 'x_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    """
        Cleaning data steps involving preprocessing and dividing data
        Args:
            data: pd.DataFrame : Ingested Data to be cleaned
        Returns: Tuple of cleaned data
    """
    try:
        logging.info("Cleaning data")
        # Initialising preprocessing strategy class
        cleaningProcess = DataCleaning(data, DataPreprocessStrategy())
        # Running the cleaning process
        cleanedData = cleaningProcess.handle_data()
        # Initialising dividing strategy class
        dividingProcess = DataCleaning(cleanedData, DataDivideStrategy())
        # Running the dividing process
        x_train, x_test, y_train, y_test = dividingProcess.handle_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error while cleaning data: {e}")
        raise e
    