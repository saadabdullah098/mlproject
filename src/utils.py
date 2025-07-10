import os 
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
import dill

def save_object(file_path, obj):
    '''
        This function opens a a file file_obj in binary write mode at the file_path 
        and saves the obj file inside it after converting it to byte stream.
        Input: file_path and the file
        Output: Saved .pkl file at the file path
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        custom_err = CustomException(e, sys)
        logging.error(custom_err)
        raise custom_err

def load_object(file_path):
    '''
        This function opens a pkl file in read byte mode and loads it to a variable
        Input: file_path to .pkl file
        Output: model loaded to a variable
    '''
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        custom_err = CustomException(e, sys)
        logging.error(custom_err)
        raise custom_err


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    '''
        This function compares various models and thier associated parameters to generates a report.
        Input: Training and testing data along with a dictionary of models & parameters
        Output: Report containing r2 scores of all the models 
    '''
    try:
        report = {}
        for i in range(len(list(models))):
            
            # Set model and its parameters
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            # Do GridSearchCV to find the best model parameter
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Select the best 
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        custom_err = CustomException(e, sys)
        logging.error(custom_err)
        raise custom_err