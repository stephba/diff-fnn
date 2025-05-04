import logging
from diff_fnn.utils import load_config
from diff_fnn.data.data import load_preprocessed_data
from diff_fnn.evaluation import evaluation
from diff_fnn.utils import get_pytorch_device, check_results_folder_exists
from diff_fnn.tables_and_visualisations import generate_tables_and_visualisations
import os
import argparse

LOG_FILE_NAME = 'main.log'
N_REPEAT_EXP = 10

if __name__ == "__main__":
    # read in config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='config file path')
    parser.add_argument('--dry-run', action='store_true', help='perform a dry run only with 1 epoch and 1 repetition of experiments')
    parser.add_argument('--only-cpu', action='store_true', help='gpu support disabled; only use cpu')
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
    if args.dry_run:
        logging.warning("Dry run activated!")
        config.training.num_of_epochs = 1
        config.evaluation.lightgcn_epochs = 1
        N_REPEAT_EXP = 2
    # create results folder if not existing
    check_results_folder_exists(config.results_path)
    # get pytorch device
    device = get_pytorch_device()
    if args.only_cpu:
        logging.warning("Device manually set to CPU!")
        device = "cpu"
    # load preprocessed data
    train_data, val_data, train_plus_val_data, test_data = load_preprocessed_data(config)
    # training and evaluation
    if config.use_final_testset:
        evaluation(config, N_REPEAT_EXP, device, train_plus_val_data, test_data)
    else:
        evaluation(config, N_REPEAT_EXP, device, train_data, val_data)
    # generate tables and visualisations
    generate_tables_and_visualisations(config, N_REPEAT_EXP)
