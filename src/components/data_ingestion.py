import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# Class to define data path variables
# Use dataclass decorator when only defining variables - it automatically generates the __init__ method
@dataclass 
class DataIngestionConfig:
    '''
        This class creates the file path for data(s)
    '''
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

# Class to initialize data ingestion
class DataIngestion:
    def __init__(self):
        '''
            This function initializes the 3 path variables and are saved inside the object of class
        '''
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
            This function initiates data ingestion.
            Input: raw data as a DataFrame on line 40
            Output: file path to saved train and test data
        '''
        logging.info("Entered the data ingestion method or component")
        try:
            # Data can be read from any source
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset")
            
            # Making the artifacts directory if it doesn't already exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data inside artifacts
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data
            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test data inside artifacts 
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return (
                # Return these paths so that data transformation can easily read these paths
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            custom_err = CustomException(e, sys)
            logging.error(custom_err)
            raise custom_err

# Initiate data ingestion        
if __name__ == '__main__':
    # First run DataIngestion to read raw data and split it into train and test
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    # Then apply DataTransformation 
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))