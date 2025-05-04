import torch
from typing import Callable
from torch_geometric.data import HeteroData
from diff_fnn.utils import Config
import numpy as np

def precision_at_k_per_user(relevancy, predictions, k):
    """Precision at k for a single user

    Args:
        relevancy (torch tensor): binary array indicating which items are relevant (sorted by itemID)
        predictions (torch tensor): probabilities for all items how likely the user likes the item (sorted by itemID)
        k (int): for selecting the top-k results
    """
    sorted_indices = torch.flip(torch.argsort(predictions), dims=[0])  # flip for descending order
    sorted_relevancy = relevancy[sorted_indices]
    top_k_relevancy = sorted_relevancy[:k]
    relevant_count = torch.sum(top_k_relevancy)
    precision_at_k = relevant_count / k
    return precision_at_k.item()

def recall_at_k_per_user(relevancy, predictions, k):
    """Recall at k for a single user

    Args:
        relevancy (torch tensor): binary array indicating which items are relevant (sorted by itemID)
        predictions (torch tensor): probabilities for all items how likely the user likes the item (sorted by itemID)
        k (int): for selecting the top-k results
    """
    sorted_indices = torch.flip(torch.argsort(predictions), dims=[0])  # flip for descending order
    sorted_relevancy = relevancy[sorted_indices]
    top_k_relevancy = sorted_relevancy[:k]
    relevant_count = torch.sum(top_k_relevancy)
    total_relevant_count = torch.sum(relevancy)
    recall_at_k = relevant_count / total_relevant_count
    return recall_at_k.item()

def average_precision_at_k_per_user(relevancy, predictions, k):
    sorted_indices = torch.flip(torch.argsort(predictions), dims=[0])  
    sorted_relevancy = relevancy[sorted_indices]
    top_k_relevancy = sorted_relevancy[:k]
    total_relevant_count = torch.sum(relevancy).item()

    if total_relevant_count == 0:
        return 0.0

    precision_sum = 0.0
    relevant_items = 0
    for i in range(min(k, len(top_k_relevancy))):
        if top_k_relevancy[i] == 1:
            relevant_items += 1
            precision_sum += relevant_items / (i + 1)

    return (precision_sum / min(total_relevant_count, k))

def dcg_at_k_per_user(relevancy, predictions, k):
    """Discounted Cumulative Gain at k for a single user

    Args:
        relevancy (torch tensor): binary array indicating which items are relevant (sorted by itemID)
        predictions (torch tensor): probabilities for all items how likely the user likes the item (sorted by itemID)
        k (int): for selecting the top-k results
    """
    sorted_indices = torch.flip(torch.argsort(predictions), dims=[0])  # flip for descending order
    sorted_relevancy = relevancy[sorted_indices]
    top_k_relevancy = sorted_relevancy[:k]

    dcg = torch.sum((top_k_relevancy.float() / torch.log2(torch.arange(2, k + 2, device=top_k_relevancy.device))).float())
    return dcg.item()

def idcg_at_k_per_user(relevancy, k):
    """Ideal Discounted Cumulative Gain at k for a single user

    Args:
        relevancy (torch tensor): binary array indicating which items are relevant (sorted by itemID)
        k (int): for selecting the top-k results
    """
    total_relevant_count = torch.sum(relevancy)

    idcg = torch.sum((1 / torch.log2(torch.arange(2, min(total_relevant_count, k) + 2))).float())
    return idcg.item()

def ndcg_at_k_per_user(relevancy, predictions, k):
    """Normalized Discounted Cumulative Gain at k for a single user

    Args:
        relevancy (torch tensor): binary array indicating which items are relevant (sorted by itemID)
        predictions (torch tensor): probabilities for all items how likely the user likes the item (sorted by itemID)
        k (int): for selecting the top-k results
    """
    # NOTE: if there are less than k items, use ndcg@n if there are n items
    num_items = predictions.shape[0]
    k = min(k, num_items)

    dcg = dcg_at_k_per_user(relevancy, predictions, k)
    idcg = idcg_at_k_per_user(relevancy, k)

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def evaluation_at_k(config: Config, predictions: torch.Tensor, relevancy: torch.Tensor, num_of_users: int, eval_edge_index: torch.Tensor):
    binary_relevancy = (relevancy >= config.data.good_rating_threshold).float()

    results = dict()
    for k in [5, 10]:
        precision_at_k = 0.0
        recall_at_k = 0.0
        ndcg_at_k = 0.0
        map_at_k = 0.0
        recall_true_users = 0
        for u in torch.arange(num_of_users):
            binary_relevancy_for_u = binary_relevancy[eval_edge_index[0] == u]
            pred_for_u = predictions[eval_edge_index[0] == u]
            precision_at_k += precision_at_k_per_user(binary_relevancy_for_u, pred_for_u, k)
            recall_at_k_for_u = recall_at_k_per_user(binary_relevancy_for_u, pred_for_u, k)
            if not np.isnan(recall_at_k_for_u):
                recall_at_k += recall_at_k_for_u
                recall_true_users += 1
            ndcg_at_k += ndcg_at_k_per_user(binary_relevancy_for_u, pred_for_u, k)
            map_at_k += average_precision_at_k_per_user(binary_relevancy_for_u, pred_for_u, k)
        precision_at_k /= num_of_users
        recall_at_k /= recall_true_users
        ndcg_at_k /= num_of_users
        map_at_k /= num_of_users
        results[f'P@{k}'] = precision_at_k
        results[f'R@{k}'] = recall_at_k
        results[f'NDCG@{k}'] = ndcg_at_k
        results[f'MAP@{k}'] = map_at_k
    return results
