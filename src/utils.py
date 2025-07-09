import os 
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Open a file file_obj in binary write model, convert obj to byte stream, and save it at file_path
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        custom_err = CustomException(e, sys)
        logging.error(custom_err)
        raise custom_err
        