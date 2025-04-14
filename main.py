import logging
from diff_fnn.utils import load_config
from diff_fnn.data.data import load_preprocessed_data
from diff_fnn.evaluation import evaluation
from diff_fnn.utils import get_pytorch_device, check_results_folder_exists
import os
import argparse

LOG_FILE_NAME = 'main.log'

DRY_RUN = False

if __name__ == "__main__":
    # read in config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='config file path')
    args = parser.parse_args()
    config_file = args.config

    # init logging
    logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE_NAME)
    ]
    )
    # load config
    config = load_config(config_file)
    if DRY_RUN:
        logging.warning("Dry run activated!")
        config.training.num_of_epochs = 1
    # create results folder if not existing
    check_results_folder_exists(config.results_path)
    # get pytorch device
    device = get_pytorch_device()
    # load preprocessed data
    train_data, val_data, train_plus_val_data, test_data = load_preprocessed_data(config)
    # training and evaluation
    if config.use_final_testset:
        evaluation(config, device, train_plus_val_data, test_data)
    else:
        evaluation(config, device, train_data, val_data)
