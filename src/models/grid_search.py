import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor


def main(filepath="./data/processed_data"):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Normalizing data')


    input_Xtrain = f"{filepath}/X_train_scaled.csv"
    input_ytrain = f"{filepath}/y_train.csv"

    X_train = import_dataset(input_Xtrain, sep=',', index_col='date')
    y_train = import_dataset(input_ytrain)['silica_concentrate']

    findBestModel(X_train, y_train)    
    

def findBestModel(X_train, y_train, output_filepath="./models"):
    # Linear regression
    grid_lr = LinearRegression()
    # RandomForest
    grid_gbr = GradientBoostingRegressor(random_state=22)
    
    # Configuration des hyperparamètres pour nos modèles

    param_grid_lr = [{
    'fit_intercept': [True, False],
    'positive': [True, False]
}]

    param_grid_gbr = [{'learning_rate': [0.01,0.02,0.03,0.04],
                  'subsample'    : [0.9, 0.5, 0.2, 0.1],
                  'n_estimators' : [100,500,1000, 1500],
                  'max_depth'    : [4,6,8,10]
                 }]


    gridcvs = {}

    # Instancier pour chaque paire de modèle et grille, un GridSearchCV 
    # l'enregistrer comme élément de gridcvs avec une clé correspondant au nom de l'algorithme utilisé
    classi = [ grid_lr, grid_gbr]

    #params = [param_grid_lr, param_grid_rf, param_grid_svc]

    params = [ param_grid_lr, param_grid_gbr]
    names = [ 'LinearRegression', 'GBR']

    for i, j, z in zip(classi, params, names):
        gridcv = GridSearchCV(i, param_grid= j, refit = True, cv=3)
        gridcvs[z] = gridcv    

    
    # transformation de notre matrice X_train en np.array
    X_train_array = np.asarray(X_train)
    # scores obtenus par validation croisée stratifiée à 3 splits
    skf = StratifiedKFold(n_splits = 3, shuffle = True)

    # Sauvegarder les scores retournés dans outer_scores, avec la même clé correspondant au nom du modèle.
    outer_scores = {}

    for i , j in gridcvs.items():
        cross_score = cross_val_score(j, X_train_array, y_train, cv = 3)
        outer_scores[i] = cross_score

    # Afficher, pour chaque modèle la moyenne des scores obtenus +/- l'écart type.    
    for i, j in outer_scores.items():
        print(i, '-  moyenne des scores :', j.mean(), ', écart type des scores :', j.std())

    bestmodel = gridcvs['GBR'].best_estimator_

    persistModel(bestmodel)


def persistModel(model, filepath= "./models/bestmodel.pkl"):
    with open(filepath,'wb') as f:
        pickle.dump(model,f)

    


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()