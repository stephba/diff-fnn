import torch_geometric.transforms as T
import copy
import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import negative_sampling
from diff_fnn.utils import Config, logging_decorator
import logging

from diff_fnn.data.preprocessing_movielens1m import preprocessing as movielens1m_preprocessing
from diff_fnn.data.preprocessing_synthetic import preprocessing as synthetic_preprocessing
from diff_fnn.data.preprocessing_anime_lowpop import preprocessing as anime_lowpop_preprocessing
from diff_fnn.data.preprocessing_anime import preprocessing as anime_preprocessing

PREPROCESSING_MODULES = {
    'movielens1m': movielens1m_preprocessing,
    'synthetic': synthetic_preprocessing,
    'anime-lowpop': anime_lowpop_preprocessing,
    'anime': anime_preprocessing,
}

@logging_decorator("Load preprocessed data")
def load_preprocessed_data(config: Config):
    dataset_name = config.data.name
    train_data, val_data, train_plus_val_data, test_data = PREPROCESSING_MODULES[dataset_name](config)
    logging.info(f"{train_data=}")
    logging.info(f"{val_data=}")
    logging.info(f"{train_plus_val_data=}")
    logging.info(f"{test_data=}")
    return train_data, val_data, train_plus_val_data, test_data
