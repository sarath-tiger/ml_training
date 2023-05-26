from my_ml.train import housing_pre_process_eda, housing_model_build
from my_ml.score import get_score_LR, final_predict, get_score_tree
from my_ml.logger import log_initialize,logging
import os
import yaml
import argparse
import pytest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("self", nargs="?", type=str)
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

if args.ml_model_path is None:
    ml_model_path = os.path.join(project_path, config["ml_model_path"])
else:
    ml_model_path = os.path.join(project_path, args.ml_model_path)
if args.split_data_path is None:
    split_data_path = os.path.join(project_path, config["split_data_path"])
else:
    split_data_path = os.path.join(project_path, args.split_data_path)

config['split_data_path'] = split_data_path
config['ml_model_path'] = ml_model_path

logging.info("in functional testing")
logging.info("files present in {}".format(config["split_data_path"]))
logging.info(','.join(os.listdir(split_data_path)))

result_eda = housing_pre_process_eda(config)

@pytest.mark.run(order=3)
def test_housing_pre_process_eda():
    assert result_eda == True

# result_model = housing_model_build(config) 

# @pytest.mark.run(order=4)
# def test_housing_model_build():
#     assert result_model == True


# result_lr = get_score_LR(config)
# if result_lr["result"]:
#     print("The score for Linear regression")
#     print(result_lr["score"])
#     logging.info("The score for linear regression model obtained")
# else:
#     logging.error("Unable to get score")


# @pytest.mark.run(order=5)
# def test_get_score_LR():
#     assert result_lr["result"] == True


# result_rf = get_score_tree(config)
# if result_rf["result"]:
#     print("The score for Random forest regression")
#     print(result_rf["score"])
#     logging.info("The score for random forest model obtained")
# else:
#     logging.error("Unable to get score")

# @pytest.mark.run(order=6)
# def test_get_score_tree():
#     assert result_rf["result"] == True


# result_final = final_predict(config)
# if result_final["result"]:
#     print("The score for Final model")
#     print(result_final["score"])
#     logging.info("The score for Final model obtained")
# else:
#     logging.error("Unable to get score")

# logging.info("Script Completed after getting score")


# @pytest.mark.run(order=7)
# def test_result_final():
#     assert result_final["result"] == True
