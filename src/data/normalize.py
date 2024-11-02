import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from check_structure import check_existing_file, check_existing_folder
import os


def main(filepath="./data/processed_data"):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Normalizing data')


    input_Xtrain = f"{filepath}/X_train.csv"
    input_Xtest = f"{filepath}/X_test.csv"

    normalize_data(input_Xtrain, 'X_train',  filepath)    
    normalize_data(input_Xtest, 'X_test',  filepath)

def normalize_data(input_file, dataset_name,  output_filepath):
    df = import_dataset(input_file, sep=',',index_col='date')

    df.info()

    # Normalize data

        

    df_scaled=(df-df.min())/(df.max()-df.min())

    output_file = f"{output_filepath}/{dataset_name}_scaled.csv"

    if check_existing_file(output_file):
            df_scaled.to_csv(output_file, index=True)

        

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)



            
            
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()