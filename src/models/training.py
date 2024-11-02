import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle


def main(filepath="./data/processed_data"):
    
    logger = logging.getLogger(__name__)
    logger.info('Training model')


    input_Xtrain = f"{filepath}/X_train_scaled.csv"
    input_ytrain = f"{filepath}/y_train.csv"

    X_train = import_dataset(input_Xtrain, sep=',', index_col='date')
    y_train = import_dataset(input_ytrain)['silica_concentrate']

    model = loadModel()

    model.fit(X_train, y_train)

    persistModel(model)


def persistModel(model, filepath= "./models/trainedmodel.pkl"):
    with open(filepath,'wb') as f:
        pickle.dump(model,f)


def loadModel(filepath= "./models/bestmodel.pkl"):
    with open(filepath,'rb') as f:
        return pickle.load(f)


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()