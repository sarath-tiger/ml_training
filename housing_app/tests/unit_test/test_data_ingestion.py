from my_ml.ingest_data import fetch_housing_data, get_train_val_test_data
from my_ml.train import housing_pre_process_eda, housing_model_build
from my_ml.score import get_score_LR, final_predict, get_score_tree
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
parser.add_argument("ml_model_path", nargs="?", type=str)
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
if args.ml_model_path is None:
    ml_model_path = os.path.join(project_path, config["ml_model_path"])
else:
    ml_model_path = os.path.join(project_path, args.ml_model_path)

config['split_data_path'] = split_data_path
config['ml_model_path'] = ml_model_path
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

result_model = housing_model_build(config) 

@pytest.mark.run(order=4)
def test_housing_model_build():
    assert result_model == True

result_lr = get_score_LR(config)
if result_lr["result"]:
    print("The score for Linear regression")
    print(result_lr["score"])
    logging.info("The score for linear regression model obtained")
else:
    logging.error("Unable to get score")


@pytest.mark.run(order=5)
def test_get_score_LR():
    assert result_lr["result"] == True


result_rf = get_score_tree(config)
if result_rf["result"]:
    print("The score for Random forest regression")
    print(result_rf["score"])
    logging.info("The score for random forest model obtained")
else:
    logging.error("Unable to get score")

@pytest.mark.run(order=6)
def test_get_score_tree():
    assert result_rf["result"] == True


result_final = final_predict(config)
if result_final["result"]:
    print("The score for Final model")
    print(result_final["score"])
    logging.info("The score for Final model obtained")
else:
    logging.error("Unable to get score")


@pytest.mark.run(order=7)
def test_result_final():
    assert result_final["result"] == True

logging.info("Script Completed after getting score")