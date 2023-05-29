import logging
import os


def log_initialize(filename):
    """
    This module is to initialize logger configuration with log path, level and format

    Args:
        filename (str): The filename with/without path to save the generated log in a file

    Returns:
        None
    """
    project_path = os.path.dirname(os.path.dirname(os.getcwd()))
    log_path = os.path.join(
        project_path, "logs", "{}.log".format(filename.split(".")[0])
    )
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
