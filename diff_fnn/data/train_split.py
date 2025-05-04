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
from diff_fnn.utils import Config
import pandas as pd
import os

def get_reverse_edge_name(edge_name: Tuple[str]):
    return (edge_name[2], 'rev_' + edge_name[1], edge_name[0])

# similiar to RandomLinkSplit
class TemporalLinkSplit(BaseTransform):
    def __init__(
        self,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.2,
        is_undirected: bool = False,
        key: str = 'edge_label',
        split_labels: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
        disjoint_train_ratio: Union[int, float] = 0.0,
        edge_types: Optional[Union[EdgeType, List[EdgeType]]] = None,
        rev_edge_types: Optional[Union[
            EdgeType,
            List[Optional[EdgeType]],
        ]] = None,
    ) -> None:
        if isinstance(edge_types, list):
            if rev_edge_types is None:
                rev_edge_types = [None] * len(edge_types)

            assert isinstance(rev_edge_types, list)
            assert len(edge_types) == len(rev_edge_types)

        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Tuple[
            Union[Data, HeteroData],
            Union[Data, HeteroData],
            Union[Data, HeteroData],
    ]:
        edge_types = self.edge_types
        rev_edge_types = self.rev_edge_types

        train_data = copy.copy(data)
        val_data = copy.copy(data)
        test_data = copy.copy(data)

        if isinstance(data, HeteroData):
            assert isinstance(train_data, HeteroData)
            assert isinstance(val_data, HeteroData)
            assert isinstance(test_data, HeteroData)

            if edge_types is None:
                raise ValueError(
                    "The 'TemporalLinkSplit' transform expects 'edge_types' to "
                    "be specified when operating on 'HeteroData' objects")
            
            if not isinstance(edge_types, list):
                assert not isinstance(rev_edge_types, list)
                edge_types = [edge_types]
                rev_edge_types = [rev_edge_types]

            stores = [data[edge_type] for edge_type in edge_types]
            train_stores = [train_data[edge_type] for edge_type in edge_types]
            val_stores = [val_data[edge_type] for edge_type in edge_types]
            test_stores = [test_data[edge_type] for edge_type in edge_types]
        else:
            raise NotImplementedError
        
        assert isinstance(rev_edge_types, list)
        for item in zip(stores, train_stores, val_stores, test_stores,
                        rev_edge_types):
            store, train_store, val_store, test_store, rev_edge_type = item

            is_undirected = self.is_undirected
            is_undirected &= not store.is_bipartite()
            is_undirected &= (rev_edge_type is None
                              or (isinstance(data, HeteroData)
                                  and store._key == data[rev_edge_type]._key))
            
            edge_index = store.edge_index
            if is_undirected:
                raise NotImplementedError
            else:
                device = edge_index.device
                # NOTE: make permutation according to timestamp of the edge
                perm = torch.argsort(store.time).to(device)

            num_val = self.num_val
            if isinstance(num_val, float):
                num_val = int(num_val * perm.numel())
            num_test = self.num_test
            if isinstance(num_test, float):
                num_test = int(num_test * perm.numel())

            num_train = perm.numel() - num_val - num_test

            if num_train <= 0:
                raise ValueError("Insufficient number of edges for training")
            
            train_edges = perm[:num_train]
            val_edges = perm[num_train:num_train + num_val]
            test_edges = perm[num_train + num_val:]
            train_val_edges = perm[:num_train + num_val]

            num_disjoint = self.disjoint_train_ratio
            if isinstance(num_disjoint, float):
                num_disjoint = int(num_disjoint * train_edges.numel())
            if num_train - num_disjoint <= 0:
                raise ValueError("Insufficient number of edges for training")

            # Create data splits:
            self._split(train_store, train_edges[num_disjoint:], is_undirected,
                        rev_edge_type)
            self._split(val_store, train_edges, is_undirected, rev_edge_type)
            self._split(test_store, train_val_edges, is_undirected,
                        rev_edge_type)
            
            # Create negative samples:
            num_neg_train = 0
            if self.add_negative_train_samples:
                if num_disjoint > 0:
                    num_neg_train = int(num_disjoint * self.neg_sampling_ratio)
                else:
                    num_neg_train = int(num_train * self.neg_sampling_ratio)
            num_neg_val = int(num_val * self.neg_sampling_ratio)
            num_neg_test = int(num_test * self.neg_sampling_ratio)

            num_neg = num_neg_train + num_neg_val + num_neg_test

            size = store.size()
            if store._key is None or store._key[0] == store._key[-1]:
                size = size[0]
            neg_edge_index = negative_sampling(edge_index, size,
                                               num_neg_samples=num_neg,
                                               method='sparse')

            # Adjust ratio if not enough negative edges exist
            if neg_edge_index.size(1) < num_neg:
                num_neg_found = neg_edge_index.size(1)
                ratio = num_neg_found / num_neg
                warnings.warn(
                    f"There are not enough negative edges to satisfy "
                    "the provided sampling ratio. The ratio will be "
                    f"adjusted to {ratio:.2f}.")
                num_neg_train = int((num_neg_train / num_neg) * num_neg_found)
                num_neg_val = int((num_neg_val / num_neg) * num_neg_found)
                num_neg_test = num_neg_found - num_neg_train - num_neg_val

            # Create labels:
            if num_disjoint > 0:
                train_edges = train_edges[:num_disjoint]
            self._create_label(
                store,
                train_edges,
                neg_edge_index[:, num_neg_val + num_neg_test:],
                out=train_store,
            )
            self._create_label(
                store,
                val_edges,
                neg_edge_index[:, :num_neg_val],
                out=val_store,
            )
            self._create_label(
                store,
                test_edges,
                neg_edge_index[:, num_neg_val:num_neg_val + num_neg_test],
                out=test_store,
            )

        return train_data, val_data, test_data

    def _split(
        self,
        store: EdgeStorage,
        index: Tensor,
        is_undirected: bool,
        rev_edge_type: Optional[EdgeType],
    ) -> EdgeStorage:

        edge_attrs = {key for key in store.keys() if store.is_edge_attr(key)}
        for key, value in store.items():
            if key == 'edge_index':
                continue

            if key in edge_attrs:
                value = value[index]
                if is_undirected:
                    value = torch.cat([value, value], dim=0)
                store[key] = value

        edge_index = store.edge_index[:, index]
        if is_undirected:
            edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
        store.edge_index = edge_index

        if rev_edge_type is not None:
            rev_store = store._parent()[rev_edge_type]
            for key in rev_store.keys():
                if key not in store:
                    del rev_store[key]  # We delete all outdated attributes.
                elif key == 'edge_index':
                    rev_store.edge_index = store.edge_index.flip([0])
                else:
                    rev_store[key] = store[key]

        return store

    def _create_label(
        self,
        store: EdgeStorage,
        index: Tensor,
        neg_edge_index: Tensor,
        out: EdgeStorage,
    ) -> EdgeStorage:

        edge_index = store.edge_index[:, index]
        edge_label_time = store.time[index]

        if hasattr(store, self.key):
            edge_label = store[self.key]
            edge_label = edge_label[index]
            # Increment labels by one. Note that there is no need to increment
            # in case no negative edges are added.
            if neg_edge_index.numel() > 0:
                assert edge_label.dtype == torch.long
                assert edge_label.size(0) == edge_index.size(1)
                edge_label.add_(1)
            if hasattr(out, self.key):
                delattr(out, self.key)
        else:
            edge_label = torch.ones(index.numel(), device=index.device)

        if neg_edge_index.numel() > 0:
            neg_edge_label = edge_label.new_zeros((neg_edge_index.size(1), ) +
                                                  edge_label.size()[1:])
            neg_edge_label_time = edge_label_time.new_zeros(neg_edge_index.size(1))

        if self.split_labels:
            out[f'pos_{self.key}'] = edge_label
            out[f'pos_{self.key}_index'] = edge_index
            out[f'pos_{self.key}_time'] = edge_label_time
            if neg_edge_index.numel() > 0:
                out[f'neg_{self.key}'] = neg_edge_label
                out[f'neg_{self.key}_index'] = neg_edge_index
                out[f'neg_{self.key}_time'] = neg_edge_label_time

        else:
            if neg_edge_index.numel() > 0:
                edge_label = torch.cat([edge_label, neg_edge_label], dim=0)
                edge_index = torch.cat([edge_index, neg_edge_index], dim=-1)
                edge_label_time = torch.cat([edge_label_time, neg_edge_label_time], dim=0)
            out[self.key] = edge_label
            out[f'{self.key}_index'] = edge_index
            out[f'{self.key}_time'] = edge_label_time

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')
    
def get_train_plus_val_data(train_data: HeteroData, val_data: HeteroData, test_data: HeteroData):
    """Get train and val data to train the model for subsequent evaluation on the final test data

    Args:
        train_data (HeteroData): Training Data
        val_data (HeteroData): Validation Data
        test_data (HeteroData): Test Data

    Returns:
        HeteroData: Combined Data
    """
    # start with the test_data, since only train and val edges are included in edge_index and time
    train_plus_val_data = test_data.clone()
    for edge_type in train_plus_val_data.edge_types:
        train_plus_val_data[edge_type].edge_label = torch.hstack((
            train_data[edge_type].edge_label,
            val_data[edge_type].edge_label
        ))
        train_plus_val_data[edge_type].edge_label_index = torch.hstack((
            train_data[edge_type].edge_label_index,
            val_data[edge_type].edge_label_index
        ))
        train_plus_val_data[edge_type].edge_label_time = torch.hstack((
            train_data[edge_type].edge_label_time,
            val_data[edge_type].edge_label_time
        ))
    return train_plus_val_data

def store_graph_data_as_csv(config: Config, graph_data:HeteroData, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rating_edge_name = config.data.rating_edge_name
    user_indices_np = graph_data[rating_edge_name].edge_label_index[0].numpy()
    item_indices_np = graph_data[rating_edge_name].edge_label_index[1].numpy()
    ratings_np = graph_data[rating_edge_name].edge_label.numpy()
    timestamps_np = graph_data[rating_edge_name].edge_label_time.numpy()
    df = pd.DataFrame({
        'userID': user_indices_np,
        'itemID': item_indices_np,
        'rating': ratings_np,
        'timestamp': timestamps_np
    })
    df = df.sort_values(by='userID')
    df.to_csv(csv_path, index=False)

def train_test_split(config: Config, data: HeteroData):
    train_data, val_data, test_data = TemporalLinkSplit(
        num_val=config.evaluation.val_size,
        num_test=config.evaluation.test_size,
        disjoint_train_ratio=0.0,
        # NOTE: no negative sampling, already done with ratings
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=[config.data.rating_edge_name]
    )(data)
    train_plus_val_data = get_train_plus_val_data(train_data, val_data, test_data)
    # Add a reverse ('item', 'rev_rates', 'user') relation for message passing:
    train_data = T.ToUndirected()(train_data)
    val_data = T.ToUndirected()(val_data)
    train_plus_val_data = T.ToUndirected()(train_plus_val_data)
    test_data = T.ToUndirected()(test_data)
    # Store the data also as csv files
    store_graph_data_as_csv(config, train_data, os.path.join('data', config.data.name, 'processed/train_ratings.csv'))
    store_graph_data_as_csv(config, val_data, os.path.join('data', config.data.name, 'processed/val_ratings.csv'))
    store_graph_data_as_csv(config, train_plus_val_data, os.path.join('data', config.data.name, 'processed/train_plus_val_ratings.csv'))
    store_graph_data_as_csv(config, test_data, os.path.join('data', config.data.name, 'processed/test_ratings.csv'))
    return train_data, val_data, train_plus_val_data, test_data
