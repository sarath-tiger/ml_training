import os


def import_check():
    try:
        from my_ml.logger import logging
        from my_ml.logger import log_initialize
        from my_ml.ingest_data import fetch_housing_data
        from my_ml.ingest_data import get_train_val_test_data
        from my_ml.train import housing_pre_process_eda
        from my_ml.train import housing_model_build
        from my_ml.score import get_score_LR
        from my_ml.score import final_predict
        from my_ml.score import get_score_tree

        logging.info("All modules imported")
        return True
    except Exception as e:
        print(str(e))
        logging.error("An exception occurred")


def test_import_check():
    result = import_check()
    assert result == True
