import pickle
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from my_ml.logger import logging, log_initialize

log_initialize(os.path.basename(__file__))


def get_score_LR(config):
    # lin_reg = pickle.load(os.path.join(config["ml_model_path"], "lin_reg.pkl"))
    logging.info("loading linear regression model")
    lin_reg = pickle.load(
        open(os.path.join(config["ml_model_path"], "lin_reg.pkl"), "rb")
    )
    housing_prepared = pd.read_csv(
        os.path.join(config["split_data_path"], "housing_prepared.csv")
    )
    housing_labels = pd.read_csv(
        os.path.join(config["split_data_path"], "housing_labels.csv")
    )
    logging.info("Getting the prediction data for LR")
    housing_predictions = lin_reg.predict(housing_prepared)
    logging.info("Calculating mse and mae score for LR")
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    logging.info("Score obtained for LR")
    return {"result": True, "score": {"lin_mae": lin_mae, "lin_rmse": lin_rmse}}


def get_score_tree(config):
    logging.info("Loading tree reg model and data for validtion")
    tree_reg = pickle.load(
        open(os.path.join(config["ml_model_path"], "tree_reg.pkl"), "rb")
    )
    housing_prepared = pd.read_csv(
        os.path.join(config["split_data_path"], "housing_prepared.csv")
    )
    housing_labels = pd.read_csv(
        os.path.join(config["split_data_path"], "housing_labels.csv")
    )
    housing_predictions = tree_reg.predict(housing_prepared)
    logging.info("Getting mse score for the model")
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    return {"result": True, "score": {"tree_rmse": tree_rmse}}


def final_predict(config):
    logging.info("Loading final model from pickle file and data for validation")
    final_model = pickle.load(
        open(os.path.join(config["ml_model_path"], "final_model.pkl"), "rb")
    )
    X_test_prepared = pd.read_csv(
        os.path.join(config["split_data_path"], "X_test_prepared.csv")
    )
    y_test = pd.read_csv(os.path.join(config["split_data_path"], "y_test.csv"))

    final_predictions = final_model.predict(X_test_prepared)
    logging.info("Getting mse score for final ml")
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    return {"result": True, "score": {"final_rmse": final_rmse}}
