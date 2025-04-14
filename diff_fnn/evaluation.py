from typing import Callable
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import LightGCN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch_geometric.data import HeteroData
from surprise import Dataset, Reader
from surprise import NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering
import copy
import logging

from diff_fnn.model import HornNetwork
from diff_fnn.utils import Config, logging_decorator, highlight_values, highlight_max, store_df_as_html_and_latex, plot_and_save_loss, store_flowchart

N_REPEAT_EXP = 10

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

    dcg = torch.sum((top_k_relevancy.float() / torch.log2(torch.arange(2, k + 2))).float())
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

def evaluation_at_k(config: Config, graph_data: HeteroData, 
                    model: Callable[[HeteroData, torch.Tensor], torch.Tensor]):
    num_of_users = graph_data['user'].num_nodes
    eval_edge_index = graph_data[config.data.rating_edge_name].edge_label_index
    test_pred = model(graph_data)

    relevancy = graph_data[config.data.rating_edge_name].edge_label
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
            pred_for_u = test_pred[eval_edge_index[0] == u]
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

@logging_decorator("Test approach")
def test_approach(config: Config, 
                  device: torch.device,
                  train_data: HeteroData, 
                  evaluation_data: HeteroData, 
                  with_neural_network, num_of_learned_atoms, num_of_horn_clauses, learning_rates, num_of_epochs, batch_size, loss_fn, l1_lambda):
    edge_name = config.data.rating_edge_name
    predefined_names = train_data[edge_name].edge_label_predefined_names
    num_of_predefined_features = len(predefined_names)
    
    results_at_k = dict()
    for i in range(N_REPEAT_EXP):
        model = HornNetwork(config, num_of_predefined_features, num_of_horn_clauses, num_of_learned_atoms, with_neural_network).to(device)
        param_group = [
            {'params': model.horn_layers.parameters(), 'lr': learning_rates['horn_layers']}
        ]
        if with_neural_network:
            param_group.append({'params': model.neural_net.parameters(), 'lr': learning_rates['neural_net']})
        optimizer = torch.optim.Adam(param_group)

        train_losses, val_losses = model.fit(train_data, evaluation_data, optimizer, loss_fn, l1_lambda, num_of_epochs)
        plot_and_save_loss(train_losses, val_losses, os.path.join(config.results_path, f"losses_run_{i}.pdf"))

        ### EVALUATION
        # for evaluation compute the body, i.e. AND(body), 
        # combine with the weights OR(body1, 1 - weight1) AND OR(body2, 1 - weight2) AND ....
        # 1-weight to get it correct, here see Survey paper (weights have contrary meanings)
        with torch.no_grad():
            model.eval()
            final_fuzzy_weights = model.get_fuzzy_weights().detach()
            if with_neural_network:
                variables_names = [f"Learned Atom {i+1}" for i in range(num_of_learned_atoms)]
                variables_names += predefined_names
            else:
                variables_names = predefined_names
            final_fuzzy_weights_df = pd.DataFrame(final_fuzzy_weights, columns=variables_names, index=[f"Rule {i+1}" for i in range(len(final_fuzzy_weights))])
            final_fuzzy_weights_styler = final_fuzzy_weights_df.style.apply(highlight_values)

            store_df_as_html_and_latex(
                final_fuzzy_weights_styler, 
                os.path.join(config.results_path, f"fuzzy_weights_run_{i}.html"),
                os.path.join(config.results_path, f"fuzzy_weights_run_{i}.tex")
            )

            # compute scores for all user-item pairs to get top-k results
            results_at_k_this_run = evaluation_at_k(config, evaluation_data, model)
            logging.info(f"{results_at_k_this_run=}")
            if i == 0:
                for key in results_at_k_this_run:
                    results_at_k[key] = [results_at_k_this_run[key]]
            else:
                for key in results_at_k_this_run:
                    results_at_k[key].append(results_at_k_this_run[key])

            # Rule extraction:
            final_fuzzy_weights_extracted_rules = model.get_fuzzy_weights(extract_rules=True).detach()
            rules = [variables_names] * final_fuzzy_weights_extracted_rules.shape[0]
            weights = np.array(final_fuzzy_weights_extracted_rules)
            rules = []
            weights = []
            for rule_weights in final_fuzzy_weights_extracted_rules:
                n = np.array(variables_names)[rule_weights > 0.0]
                w = rule_weights[rule_weights > 0.0]
                sort_indices = torch.flip(torch.argsort(w), dims=[0])  # sort by weights decreasing
                n = np.atleast_1d(n[sort_indices])
                w = w[sort_indices]
                rules.append(n)
                weights.append(np.array(w))
            store_flowchart(rules, weights, os.path.join(config.results_path, f"extracted_rules_run_{i}.pdf"))

    return {
        key: results_at_k[key]
        for key in results_at_k
    }

@logging_decorator("Test all ones baseline")
def test_baseline_all_ones(config: Config, evaluation_data: HeteroData):
    def model(graph_data: HeteroData):
        edge_name = config.data.rating_edge_name
        edge_index = graph_data[edge_name].edge_label_index
        return torch.ones_like(edge_index[0])
    
    # compute scores for all user-item pairs to get top-k results
    results_at_k = evaluation_at_k(config, evaluation_data, model)
    logging.info(f"{results_at_k=}")

    # no standard deviation for this baseline
    return {
        key: results_at_k[key]
        for key in results_at_k
    }

@logging_decorator("Test all zeros baseline")
def test_baseline_all_zeros(config: Config, evaluation_data: HeteroData):
    def model(graph_data: HeteroData):
        edge_name = config.data.rating_edge_name
        edge_index = graph_data[edge_name].edge_label_index
        return torch.zeros_like(edge_index[0])

    # compute scores for all user-item pairs to get top-k results
    results_at_k = evaluation_at_k(config, evaluation_data, model)
    logging.info(f"{results_at_k=}")
    # no standard deviation for this baseline
    return {
        key: results_at_k[key]
        for key in results_at_k
    }

@logging_decorator("Test decision tree baseline")
def test_baseline_decision_tree(config: Config, train_data: HeteroData, evaluation_data: HeteroData, use_raw: bool):
    if not use_raw:
        raise NotImplementedError
    
    item_name = config.data.rating_edge_name[-1]
    
    results_at_k = dict()
    for i in range(N_REPEAT_EXP):
        decision_tree_baseline = DecisionTreeClassifier(random_state=42)

        y_train = (train_data[config.data.rating_edge_name].edge_label >= config.data.good_rating_threshold).float()
        X_train = torch.cat(
            (train_data['user']['x'][train_data[config.data.rating_edge_name].edge_index[0]],
            train_data[item_name]['x'][train_data[config.data.rating_edge_name].edge_index[1]]), 
            dim=1)

        decision_tree_baseline.fit(X_train, y_train)

        def model(graph_data: HeteroData):
            edge_name = config.data.rating_edge_name
            edge_index = graph_data[edge_name].edge_label_index
            X_evaluation = torch.cat(
                (graph_data['user']['x'][edge_index[0]],
                graph_data[item_name]['x'][edge_index[1]]), 
                dim=1)
            return torch.Tensor(decision_tree_baseline.predict(X_evaluation))

        # compute scores for all user-item pairs to get top-k results
        results_at_k_this_run = evaluation_at_k(config, evaluation_data, model)
        logging.info(f"{results_at_k_this_run=}")
        if i == 0:
            for key in results_at_k_this_run:
                results_at_k[key] = [results_at_k_this_run[key]]
        else:
            for key in results_at_k_this_run:
                results_at_k[key].append(results_at_k_this_run[key])

    return {
        key: results_at_k[key]
        for key in results_at_k
    }

@logging_decorator("Test random forest baseline")
def test_baseline_random_forest(config: Config, train_data: HeteroData, evaluation_data: HeteroData, use_raw: bool):
    if not use_raw:
        raise NotImplementedError
    
    item_name = config.data.rating_edge_name[-1]
    
    results_at_k = dict()
    for i in range(N_REPEAT_EXP):
        random_forest_baseline = RandomForestClassifier(random_state=42)

        y_train = (train_data[config.data.rating_edge_name].edge_label >= config.data.good_rating_threshold).float()
        X_train = torch.cat(
            (train_data['user']['x'][train_data[config.data.rating_edge_name].edge_index[0]],
            train_data[item_name]['x'][train_data[config.data.rating_edge_name].edge_index[1]]), 
            dim=1)
        
        random_forest_baseline.fit(X_train, y_train)

        def model(graph_data: HeteroData):
            edge_name = config.data.rating_edge_name
            edge_index = graph_data[edge_name].edge_label_index
            X_evaluation = torch.cat(
                (graph_data['user']['x'][edge_index[0]],
                graph_data[item_name]['x'][edge_index[1]]), 
                dim=1)
            return torch.Tensor(random_forest_baseline.predict(X_evaluation))

        # compute scores for all user-item pairs to get top-k results
        results_at_k_this_run = evaluation_at_k(config, evaluation_data, model)
        logging.info(f"{results_at_k_this_run=}")
        if i == 0:
            for key in results_at_k_this_run:
                results_at_k[key] = [results_at_k_this_run[key]]
        else:
            for key in results_at_k_this_run:
                results_at_k[key].append(results_at_k_this_run[key])

    return {
        key: results_at_k[key]
        for key in results_at_k
    }

# see also https://github.com/pyg-team/pytorch_geometric/blob/master/examples/lightgcn.py
@logging_decorator("Test light gcn baseline")
def test_baseline_light_gcn(config: Config, 
                            device: torch.device,
                            train_data: HeteroData, 
                            evaluation_data: HeteroData, 
                            use_raw: bool):
    if not use_raw:
        raise NotImplementedError
    
    batch_size = 8192
    edge_name = config.data.rating_edge_name
    item_name = edge_name[-1]
    train_edge_index = train_data[edge_name].edge_index
    ratings = train_data[edge_name].edge_label

    results_at_k = dict()
    for i in range(N_REPEAT_EXP):
        num_users = train_data['user'].num_nodes
        num_items = train_data[item_name].num_nodes

        # Use all message passing edges as training labels
        train_loader = torch.utils.data.DataLoader(
            range(train_edge_index.size(1)),
            shuffle=True,
            batch_size=batch_size,
        )

        model = LightGCN(
            num_nodes=train_data.num_nodes,
            embedding_dim=64,
            num_layers=2,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.evaluation.lightgcn_lr)

        def train():
            total_loss = total_examples = 0

            for index in train_loader:
                batch_edge_label_index = train_edge_index[:, index]
                batch_ratings = ratings[index]
                pos_edge_label_index = batch_edge_label_index[:, batch_ratings >= config.data.good_rating_threshold]
                neg_edge_label_index = batch_edge_label_index[:, batch_ratings < config.data.good_rating_threshold]
                # positive and negative examples must have same size -> truncate longer set
                min_size = min(pos_edge_label_index.shape[1], neg_edge_label_index.shape[1])
                pos_edge_label_index = pos_edge_label_index[:, :min_size]
                neg_edge_label_index = neg_edge_label_index[:, :min_size]

                optimizer.zero_grad()
                pos_rank = model(train_edge_index, pos_edge_label_index)
                neg_rank = model(train_edge_index, neg_edge_label_index)

                loss = model.recommendation_loss(
                    pos_rank,
                    neg_rank,
                    node_id=batch_edge_label_index.unique(),
                )
                loss.backward()
                optimizer.step()

                total_loss += float(loss) * pos_rank.numel()
                total_examples += pos_rank.numel()

            return total_loss / total_examples

        # training
        losses = []
        for epoch in tqdm(range(1, 51)):
            loss = train()
            losses.append(loss)
        # TODO: also plot here evaluation loss
        plot_and_save_loss(losses, losses, os.path.join(config.results_path, f"light_gcn_losses_run_{i}.pdf"))

        # evaluation
        with torch.no_grad():
            model.eval()
            def model_fun(graph_data: HeteroData):
                edge_name = config.data.rating_edge_name
                edge_index = graph_data[edge_name].edge_label_index
                emb = model.get_embedding(edge_index)
                user_emb, book_emb = emb[:num_users], emb[num_users:]

                result = (user_emb @ book_emb.t())[edge_index[0], edge_index[1]]
                return result

            # compute scores for all user-item pairs to get top-k results
            results_at_k_this_run = evaluation_at_k(config, evaluation_data, model_fun)
            logging.info(f"{results_at_k_this_run=}")
            if i == 0:
                for key in results_at_k_this_run:
                    results_at_k[key] = [results_at_k_this_run[key]]
            else:
                for key in results_at_k_this_run:
                    results_at_k[key].append(results_at_k_this_run[key])

    return {
        key: results_at_k[key]
        for key in results_at_k
    }

def get_surprise_trainset(config: Config, graph_data: HeteroData):
    edge_name = config.data.rating_edge_name
    # see also https://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset
    ratings = graph_data[edge_name].edge_label
    userIDs = graph_data[edge_name].edge_label_index[0]
    itemIDs = graph_data[edge_name].edge_label_index[1]
    # Creation of the dataframe. Column names are irrelevant.
    ratings_dict = {
        "itemID": itemIDs,
        "userID": userIDs,
        "rating": ratings,
    }
    df = pd.DataFrame(ratings_dict)
    # A reader is still needed but only the rating_scale param is required.
    reader = Reader(rating_scale=(1, 5))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
    return data.build_full_trainset()

def get_surprise_testset(edge_index):
    # only depend on edge_index, do not need graph

    ratings_dict = {
        "itemID": edge_index[1],
        "userID": edge_index[0],
        "rating": - torch.ones_like(edge_index[0]),  # do not know correct ratings for testset
    }
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
    return data.construct_testset(data.raw_ratings)

@logging_decorator("Test surprise baselines")
def test_baseline_surprise(config: Config, train_data: HeteroData, evaluation_data: HeteroData):
    trainset = get_surprise_trainset(config, train_data)

    results_at_k = dict()
    algorithms = {
            "NormalPredictor": NormalPredictor,
            "BaselineOnly": BaselineOnly,
            "KNNBasic": KNNBasic,
            # "KNNWithMeans": KNNWithMeans,
            # "KNNWithZScore": KNNWithZScore,
            # "KNNBaseline": KNNBaseline,
            "SVD": SVD,
            # "SVDpp": SVDpp,
            # "NMF": NMF,
            # "SlopeOne": SlopeOne,
            # "CoClustering": CoClustering
        }

    for name, algo_class in algorithms.items():
        logging.info(f"Test {name} baseline...")
        if name not in results_at_k:
            results_at_k[name] = {}
        for i in range(N_REPEAT_EXP):
            algo = algo_class()

            # training
            algo.fit(trainset)

            # evaluation
            def model(graph_data: HeteroData):
                edge_name = config.data.rating_edge_name
                edge_index = graph_data[edge_name].edge_label_index
                testset = get_surprise_testset(edge_index)
                predictions = np.array(algo.test(testset))[:, 3].astype(float)
                return torch.Tensor(predictions)
            # compute scores for all user-item pairs to get top-k results
            results_at_k_this_run = evaluation_at_k(config, evaluation_data, model)
            logging.info(f"{results_at_k_this_run=}")
            for key in results_at_k_this_run:
                if key not in results_at_k[name]:
                    results_at_k[name][key] = []
                results_at_k[name][key].append(results_at_k_this_run[key])

    return {
        algo: {
            key: results_at_k[algo][key]
            for key in results_at_k[algo]
        } for algo in results_at_k
    }

@logging_decorator("Evaluate approach and baselines")
def evaluation(config: Config, 
               device: torch.device,
               train_data: HeteroData, 
               evaluation_data: HeteroData):
    with_neural_network = config.model.with_neural_network
    num_of_learned_atoms = config.model.num_of_learned_atoms
    num_of_horn_clauses = config.model.num_of_horn_clauses
    learning_rates = config.training.learning_rates
    num_of_epochs = config.training.num_of_epochs
    batch_size = config.training.batch_size
    loss_fn = config.training.loss_fn
    l1_lambda = config.training.l1_lambda
    
    approach_results = test_approach(config, device, train_data, evaluation_data, with_neural_network, num_of_learned_atoms, num_of_horn_clauses, learning_rates, num_of_epochs, batch_size, loss_fn, l1_lambda)
    surprise_results = test_baseline_surprise(config, train_data, evaluation_data)
    # gnn_raw_results = test_baseline_gnn(config, train_data, evaluation_data, use_raw=True)
    light_gcn_raw_results = test_baseline_light_gcn(config, device, train_data, evaluation_data, use_raw=True)

    predefined_atoms_available = False  # TEMP

    if predefined_atoms_available:
        decision_tree_predefined_results = test_baseline_decision_tree(config, train_data, evaluation_data, use_raw=False)
        random_forest_predefined_results = test_baseline_random_forest(config, train_data, evaluation_data, use_raw=False)

    # Organize results into a dictionary
    results_dict = {
        'Our Approach': approach_results,
        # 'Baseline All Ones': test_baseline_all_ones(config, evaluation_data),
        # 'Baseline All Zeros': test_baseline_all_zeros(config, evaluation_data),
        # 'Decision Tree Raw': test_baseline_decision_tree(config, train_data, evaluation_data, use_raw=True),
        # 'Random Forest Raw': test_baseline_random_forest(config, train_data, evaluation_data, use_raw=True),
        # 'GNN': gnn_raw_results,
    }
    for algo in surprise_results:
        results_dict[f'{algo}'] = surprise_results[algo]
    if predefined_atoms_available:
        results_dict['Decision Tree Predefined'] = decision_tree_predefined_results
        results_dict['Random Forest Predefined'] = random_forest_predefined_results

    results_dict['LightGCN'] = light_gcn_raw_results

    # store raw results
    raw_results_df = pd.DataFrame({
        _approach_: {_metric_: _values_ for _metric_, _values_ in _results_.items()}
        for _approach_, _results_ in results_dict.items()
    }).transpose()
    raw_results_df.to_csv(os.path.join(config.results_path, "raw_results.csv"))

    # Convert dictionary to DataFrame
    results_df = pd.DataFrame({
        _approach_: {_metric_: f"{np.mean(_values_):.3f} ± {np.std(_values_):.3f}" for _metric_, _values_ in _results_.items()}
        for _approach_, _results_ in results_dict.items()
    }).transpose()

    # Mark the best models with a star
    for column in results_df.columns:
        max_value = results_df[column].apply(lambda x: float(x.split(' ±')[0])).max()
        results_df[column] = results_df[column].apply(lambda x: x + ' *' if float(x.split(' ±')[0]) == max_value else x)

    results_styler = results_df.style.apply(highlight_max)

    store_df_as_html_and_latex(
        results_styler, 
        os.path.join(config.results_path, "results.html"),
        os.path.join(config.results_path, "results.tex"),
    )

    return results_df

