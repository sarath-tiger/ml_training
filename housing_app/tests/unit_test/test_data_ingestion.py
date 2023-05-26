from my_ml.ingest_data import fetch_housing_data, get_train_val_test_data
from my_ml.train import housing_pre_process_eda, housing_model_build
from my_ml.logger import logging, log_initialize
import os
import yaml
import argparse
import pytest
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("self", nargs="?", type=str)
parser.add_argument("dataset_path", nargs="?", type=str)
parser.add_argument("split_data_path", nargs="?", type=str)
args = parser.parse_args()


log_initialize(os.path.basename(__file__))
project_path = os.path.join((os.getcwd().split("housing_app")[0]), "housing_app")
# project_path = os.path.dirname(os.path.dirname(os.path.basename(__file__)))
config_file = os.path.join(project_path, "config", "housing.yml")

print(config_file)
with open(config_file, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

git_url = config["git_url"]
if args.dataset_path is None:
    dataset_path = os.path.join(project_path, config["dataset_out_path"])
else:
    dataset_path = os.path.join(project_path, args.dataset_path)
if args.split_data_path is None:
    split_data_path = os.path.join(project_path, config["split_data_path"])
else:
    split_data_path = os.path.join(project_path, args.split_data_path)
config['split_data_path'] = split_data_path
data_result = fetch_housing_data(git_url, dataset_path)
if data_result:
    logging.info("Data has been downloaded in {}".format(dataset_path))
else:
    logging.error("Data download has failed")

@pytest.mark.run(order=1)
def test_fetch_housing_data():
    assert data_result == True


split_result = get_train_val_test_data(dataset_path, split_data_path)
if split_result:
    logging.info("Data has been splitted and saved in {}".format(split_data_path))
    logging.info("file present in {}".format(split_data_path))
    logging.info(','.join(os.listdir(split_data_path)))
    logging.info("Exiting the split function")
    logging.info("******************")
    logging.info(config['split_data_path'])
    logging.info("Ingest data script testing completed..!")
else:
    logging.error("Data splitting failed")

@pytest.mark.run(order=2)
def test_get_train_val_test_data():
    assert split_result == True

result_eda = housing_pre_process_eda(config)

@pytest.mark.run(order=3)
def test_housing_pre_process_eda():
    assert result_eda == True
