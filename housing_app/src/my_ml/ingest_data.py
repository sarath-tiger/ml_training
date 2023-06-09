import os
import tarfile
import pandas as pd
import numpy as np
from six.moves import urllib
from my_ml.logger import log_initialize, logging
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

log_initialize(os.path.basename(__file__))


# Calling function to download data
def fetch_housing_data(git_url, data_path):
    """
    This module is to download the data file from the given git hub url

    Args:
        git_url (str): This is the url from which data file can be downloaded
        data_path (str): This is the path to store the downloaded data file

    Returns:
        Boolean
    """
    os.makedirs(data_path, exist_ok=True)
    tgz_path = os.path.join(data_path, "housing.tgz")
    urllib.request.urlretrieve(git_url, tgz_path)
    logging.info("Data downloaded from {}".format(git_url))
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=data_path)
    logging.info("data unzipped")
    housing_tgz.close()
    return True


def load_housing_data(data_path):
    """
    This module is to load the csv data to a dataframe

    Args:
        data_path (str): The path containing the csv file

    Returns:
        Dataframe of csv data
    """
    csv_path = os.path.join(data_path, "housing.csv")
    logging.info("Convering housing to dataframe")
    return pd.read_csv(csv_path)


def get_train_val_test_data(data_path, split_data_path):
    """
    This module splits the given dataframe into train and test data set

    Args:
        data_path (str): The path that contains source csv file
        split_data_path (str): The path where the splitted csv data has to be stored

    Returns:
        Boolean
    """
    logging.info("Started splitting the data")
    os.makedirs(split_data_path, exist_ok=True)
    housing = load_housing_data(data_path)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    strat_train_set.to_csv(
        os.path.join(split_data_path, "strat_train_set.csv"), index=False
    )
    logging.info("strat_train_set saved as csv")
    strat_test_set.to_csv(
        os.path.join(split_data_path, "strat_test_set.csv"), index=False
    )
    logging.info("strat_test_set saved as csv")
    train_set.to_csv(os.path.join(split_data_path, "train_set.csv"), index=False)
    logging.info("train_set saved as csv")
    test_set.to_csv(os.path.join(split_data_path, "test_set.csv"), index=False)
    logging.info("test_set saved as csv")
    housing.to_csv(os.path.join(split_data_path, "housing.csv"), index=False)
    logging.info("housing saved as csv")
    return True
