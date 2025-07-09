import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
        This class creates the file path for the preprocessor
    '''
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        '''
            This function initializes the file path for the preprocessor
        '''
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
            This function imputes, encodes and scales both numerical and categorical data.
        '''
        try:
            # Separate categorical and numerical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Create numerical pipeline to handle missing data and scale values
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Create categorical pipeline to handle missing data, one-hot-encode, and scale values
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')

            # Apply pipelines on each datatype 
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            # Return the preprocessed data
            return preprocessor
        
        except Exception as e:
            custom_err = CustomException(e, sys)
            logging.error(custom_err)
            raise custom_err
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
            This function initiates data transformation.
            Input: train and test data path (ie. output of data_ingestion.py)
            Output: Arrays of the train and test transformed data along with the trained preprocessor.pkl file.
        '''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test datasets sucessfully read.')

            logging.info('Obtaining preprocessing object.')
            preprocessing_obj = self.get_data_transformer_obj()

            #Change this to the desired target column
            target_column_name = 'math_score'

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saving preprocessed objects")
            # save_object() function is in utils.py
            save_object(
                # Calls the preprocessor_obj_file_path from the DataTransformationConfig class
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                # Saves the trained preprocessing_obj that was initialized in this instance
                obj = preprocessing_obj
            )

            # Return the transformed data along with the preprocessor.pkl file 
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            custom_err = CustomException(e, sys)
            logging.error(custom_err)
            raise custom_err

