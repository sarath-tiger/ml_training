import pickle
import os
import numpy as np
import yaml
import pandas as pd
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from my_ml.logger import logging, log_initialize

log_initialize(os.path.basename(__file__))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("split_data_path", nargs="?", type=str)
parser.add_argument("ml_model_path", nargs="?", type=str)
args = parser.parse_args()


def get_score_LR(config):
    """
    This module is to get the score of Linear regression model with validation data set

    Args:
        config (dictionary): A dictonary containing keys ml_model_path (pickle file path) & split_data_path
        (path containing test data set)

    Returns (dictionary):
        result: Boolean
        score: {"lin_mae": lin_mae, "lin_rmse": lin_rmse}
    """
    project_path = os.path.join((os.getcwd().split("housing_app")[0]), "housing_app")

    if args.split_data_path is None:
        config["split_data_path"] = os.path.join(
            project_path, config["split_data_path"]
        )
    else:
        config["split_data_path"] = os.path.join(project_path, args.split_data_path)
    if args.ml_model_path is None:
        config["ml_model_path"] = os.path.join(project_path, config["ml_model_path"])
    else:
        config["ml_model_path"] = os.path.join(project_path, args.ml_model_path)
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
    """
    This module is to get the score of Decision Tree Regressor model with validation data set

    Args:
        config (dictionary): A dictonary containing keys ml_model_path (pickle file path) & split_data_path
        (path containing test data set)

    Returns (dictionary):
        result: Boolean
        score: {"tree_rmse": tree_rmse}
    """
    project_path = os.path.join((os.getcwd().split("housing_app")[0]), "housing_app")

    if args.split_data_path is None:
        config["split_data_path"] = os.path.join(
            project_path, config["split_data_path"]
        )
    else:
        config["split_data_path"] = os.path.join(project_path, args.split_data_path)
    if args.ml_model_path is None:
        config["ml_model_path"] = os.path.join(project_path, config["ml_model_path"])
    else:
        config["ml_model_path"] = os.path.join(project_path, args.ml_model_path)
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
    """
    This module is to get the score of Final model (best estimator) with validation data set

    Args:
        config (dictionary): A dictonary containing keys ml_model_path (pickle file path) & split_data_path
        (path containing test data set)

    Returns (dictionary):
        result: Boolean
        score: {"final_rmse": final_rmse}
    """
    project_path = os.path.join((os.getcwd().split("housing_app")[0]), "housing_app")

    if args.split_data_path is None:
        config["split_data_path"] = os.path.join(
            project_path, config["split_data_path"]
        )
    else:
        config["split_data_path"] = os.path.join(project_path, args.split_data_path)
    if args.ml_model_path is None:
        config["ml_model_path"] = os.path.join(project_path, config["ml_model_path"])
    else:
        config["ml_model_path"] = os.path.join(project_path, args.ml_model_path)
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


if __name__ == "__main__":
    project_path = os.path.join((os.getcwd().split("housing_app")[0]), "housing_app")
    config_file = os.path.join(project_path, "config", "housing.yml")

    with open(config_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(get_score_LR(config))
    print(get_score_tree(config))
    print(final_predict(config))
