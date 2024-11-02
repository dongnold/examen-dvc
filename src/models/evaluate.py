import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def main(project_dir, filepath="./data/processed_data"):
    
    logger = logging.getLogger(__name__)
    logger.info('Evaluating model')


    input_Xtest = f"{filepath}/X_test_scaled.csv"
    input_ytest = f"{filepath}/y_test.csv"

    X_test = import_dataset(input_Xtest, sep=',', index_col='date')
    y_test = import_dataset(input_ytest)['silica_concentrate']

    model = loadModel()

    predictions = model.predict(X_test)

    df_pred = pd.DataFrame(data=predictions, columns=['silica_concentrate'],index=None)

    df_pred.to_csv("./data/prediction.csv")

    mae= mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)


    metrics = {"RMSE":rmse, "MAE":mae, "r2":r2}
    
    accuracy_path = project_dir / "metrics/scores.json"

    accuracy_path.write_text(json.dumps(metrics))

def loadModel(filepath= "./models/trainedmodel.pkl"):
    with open(filepath,'rb') as f:
        return pickle.load(f)


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main(project_dir)