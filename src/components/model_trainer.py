import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    '''
    This class creates the file path for the model
    '''
    trained_model_file_path:str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        '''
            This function initializes the path variable for the model
        '''
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        '''
            This function initiates data transformation.
            Input: Arrays of test and train data (ie. Output of data_transformation.py)
            Output: Model saved as model.pkl
        '''
        try:
            logging.info('Splitting Train and Test data')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Models to be compared
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            # Training and comparing models
            logging.info('Analyzing and comparing model performances')
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, 
                                                X_test=X_test, y_test=y_test, 
                                                models=models)

            logging.info('Finding the best performing model')
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                custom_err = CustomException('No best model found - all models have r^2 below 0.6')
                logging.error(custom_err)
                raise custom_err
            
            logging.info(f'The best performing model is: {best_model_name}')
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f'The best performing model r^2 value is: {r2_square}')

            return r2_square
            
        except Exception as e:
            custom_err = CustomException(e, sys)
            logging.error(custom_err)
            raise custom_err


    