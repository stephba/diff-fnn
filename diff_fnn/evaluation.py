from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import LightGCN
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from torch_geometric.data import HeteroData
from surprise import Dataset, Reader
from surprise import NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering
import logging
import tempfile
import joblib

from diff_fnn.model import HornNetwork, HornNetworkArgs, ProductTNormOperators
from diff_fnn.utils import Config, logging_decorator, plot_and_save_loss, store_flowchart
from diff_fnn.metrics import evaluation_at_k

@logging_decorator("Test approach")
def test_approach(config: Config, 
                  n_repeat_exp,
                  device: torch.device,
                  train_data: HeteroData, 
                  evaluation_data: HeteroData, 
                  with_neural_network, num_of_learned_atoms, num_of_horn_clauses, learning_rates, num_of_epochs, batch_size, loss_fn, l1_lambda):
    edge_name = config.data.rating_edge_name
    predefined_names = train_data[edge_name].edge_label_predefined_names
    num_of_predefined_features = len(predefined_names)

    results_at_k = dict()
    for i in range(n_repeat_exp):
        args = HornNetworkArgs(
            fuzzy_operators=ProductTNormOperators,
            config=config,
            predefined_input_size=num_of_predefined_features,
            num_of_horn_clauses=num_of_horn_clauses,
            num_learned_atoms=num_of_learned_atoms,
            with_neural_net=with_neural_network
        )
        # move data to device
        train_data = train_data.to(device)
        evaluation_data = evaluation_data.to(device)

        model = HornNetwork(args).to(device)

        train_losses, val_losses = model.fit(train_data, evaluation_data, learning_rates, loss_fn, l1_lambda, num_of_epochs)
        plot_and_save_loss(train_losses, val_losses, os.path.join(config.results_path, f"losses_run_{i}.pdf"))

        # move data back to cpu
        train_data = train_data.cpu()
        evaluation_data = evaluation_data.cpu()
        model = model.cpu()

        ### EVALUATION
        with torch.no_grad():
            model.eval()

            # store fuzzy weights
            final_fuzzy_weights = model.get_fuzzy_weights().detach().cpu()
            if with_neural_network:
                variables_names = [f"Learned Atom {i+1}" for i in range(num_of_learned_atoms)]
                variables_names += predefined_names
            else:
                variables_names = predefined_names
            final_fuzzy_weights_df = pd.DataFrame(final_fuzzy_weights, columns=variables_names, index=[f"Rule {i+1}" for i in range(len(final_fuzzy_weights))])
            final_fuzzy_weights_df.to_csv(os.path.join(config.results_path, f"fuzzy_weights_run_{i}.csv"))

            # store the model
            torch.save(model, os.path.join(config.results_path, f"model_run_{i}.pth"))

            # compute scores for all user-item pairs to get top-k results
            results_at_k_this_run = evaluation_at_k(config, model(evaluation_data), evaluation_data[config.data.rating_edge_name].edge_label, evaluation_data['user'].num_nodes, evaluation_data[config.data.rating_edge_name].edge_label_index)
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

@logging_decorator("Test all ones baseline")
def test_baseline_all_ones(config: Config, n_repeat_exp, train_data: HeteroData, evaluation_data: HeteroData):
    def model(graph_data: HeteroData):
        edge_name = config.data.rating_edge_name
        edge_index = graph_data[edge_name].edge_label_index
        return torch.ones_like(edge_index[0])
    
    # compute scores for all user-item pairs to get top-k results
    results_at_k = evaluation_at_k(config, model(evaluation_data), evaluation_data[config.data.rating_edge_name].edge_label, evaluation_data['user'].num_nodes, evaluation_data[config.data.rating_edge_name].edge_label_index)
    logging.info(f"{results_at_k=}")

    # no standard deviation for this baseline
    return {
        key: results_at_k[key]
        for key in results_at_k
    }

@logging_decorator("Test all zeros baseline")
def test_baseline_all_zeros(config: Config, n_repeat_exp, train_data: HeteroData, evaluation_data: HeteroData):
    def model(graph_data: HeteroData):
        edge_name = config.data.rating_edge_name
        edge_index = graph_data[edge_name].edge_label_index
        return torch.zeros_like(edge_index[0])

    # compute scores for all user-item pairs to get top-k results
    results_at_k = evaluation_at_k(config, model(evaluation_data), evaluation_data[config.data.rating_edge_name].edge_label, evaluation_data['user'].num_nodes, evaluation_data[config.data.rating_edge_name].edge_label_index)
    logging.info(f"{results_at_k=}")
    # no standard deviation for this baseline
    return {
        key: results_at_k[key]
        for key in results_at_k
    }

@logging_decorator("Test decision tree baseline")
def test_baseline_decision_tree(config: Config, n_repeat_exp, train_data: HeteroData, evaluation_data: HeteroData, use_raw: bool):
    rating_edge_name = config.data.rating_edge_name
    item_name = rating_edge_name[-1]

    if use_raw:
        X_train = torch.cat(
            (train_data['user']['x'][train_data[config.data.rating_edge_name].edge_index[0]],
            train_data[item_name]['x'][train_data[config.data.rating_edge_name].edge_index[1]]), 
            dim=1)
    else:
        X_train = train_data[rating_edge_name]['edge_label_predefined']

    y_train = (train_data[config.data.rating_edge_name].edge_label >= config.data.good_rating_threshold).float()

    results_at_k = dict()
    for i in range(n_repeat_exp):
        decision_tree_baseline = DecisionTreeClassifier()

        decision_tree_baseline.fit(X_train, y_train)

        def model(graph_data: HeteroData):
            edge_name = config.data.rating_edge_name
            edge_index = graph_data[edge_name].edge_label_index
            if use_raw:
                X_evaluation = torch.cat(
                    (graph_data['user']['x'][edge_index[0]],
                    graph_data[item_name]['x'][edge_index[1]]), 
                    dim=1)
            else:
                X_evaluation = graph_data[rating_edge_name]['edge_label_predefined']
            return torch.Tensor(decision_tree_baseline.predict(X_evaluation))

        # compute scores for all user-item pairs to get top-k results
        results_at_k_this_run = evaluation_at_k(config, model(evaluation_data), evaluation_data[config.data.rating_edge_name].edge_label, evaluation_data['user'].num_nodes, evaluation_data[config.data.rating_edge_name].edge_label_index)
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
def test_baseline_random_forest(config: Config, n_repeat_exp, train_data: HeteroData, evaluation_data: HeteroData, use_raw: bool):
    if not use_raw:
        raise NotImplementedError
    
    item_name = config.data.rating_edge_name[-1]
    
    results_at_k = dict()
    for i in range(n_repeat_exp):
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
        results_at_k_this_run = evaluation_at_k(config, model(evaluation_data), evaluation_data[config.data.rating_edge_name].edge_label, evaluation_data['user'].num_nodes, evaluation_data[config.data.rating_edge_name].edge_label_index)
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
                            n_repeat_exp,
                            train_data: HeteroData, 
                            evaluation_data: HeteroData, 
                            device: torch.device,
                            use_raw: bool):
    if not use_raw:
        raise NotImplementedError
    
    batch_size = 8192
    edge_name = config.data.rating_edge_name
    item_name = edge_name[-1]
    train_edge_index = train_data[edge_name].edge_index
    ratings = train_data[edge_name].edge_label

    results_at_k = dict()
    for i in range(n_repeat_exp):
        num_users = train_data['user'].num_nodes
        num_items = train_data[item_name].num_nodes

        # Use all message passing edges as training labels
        train_loader = torch.utils.data.DataLoader(
            range(train_edge_index.size(1)),
            shuffle=True,
            batch_size=batch_size,
        )

        # move data to device
        train_data = train_data.to(device)
        evaluation_data = evaluation_data.to(device)

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
        for epoch in tqdm(range(0, config.evaluation.lightgcn_epochs)):
            loss = train()
            losses.append(loss)
        # TODO: also plot here evaluation loss
        plot_and_save_loss(losses, losses, os.path.join(config.results_path, f"light_gcn_losses_run_{i}.pdf"))

        # move data back to cpu
        train_data = train_data.cpu()
        evaluation_data = evaluation_data.cpu()
        model = model.cpu()

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
def test_baseline_surprise(config: Config, n_repeat_exp, train_data: HeteroData, evaluation_data: HeteroData):
    trainset = get_surprise_trainset(config, train_data)

    results_at_k = dict()
    algorithms = {
            "NormalPredictor": NormalPredictor,
            "BaselineOnly": BaselineOnly,
            "KNNBasic": KNNBasic,
            "SVD": SVD,
        }

    for name, algo_class in algorithms.items():
        logging.info(f"Test {name} baseline...")
        if name not in results_at_k:
            results_at_k[name] = {}
        for i in range(n_repeat_exp):
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
            results_at_k_this_run = evaluation_at_k(config, model(evaluation_data), evaluation_data[config.data.rating_edge_name].edge_label, evaluation_data['user'].num_nodes, evaluation_data[config.data.rating_edge_name].edge_label_index)
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

jvm_started = False
@logging_decorator("Test JRip baseline")
def test_baseline_jrip(config: Config, n_repeat_exp, train_data: HeteroData, evaluation_data: HeteroData, use_raw: bool):
    global jvm_started
    # NOTE: imports are here, since otherwise this module would affect the root logger...
    # TODO: should be moved to an extra file
    import weka.core.jvm as jvm
    from weka.classifiers import Classifier
    from weka.core.converters import Loader

    def graph_data_to_arff(config: Config, graph_data: HeteroData, use_raw, relation_name='relation'):
        rating_edge_name = config.data.rating_edge_name
        item_name = rating_edge_name[-1]

        edge_label_index = graph_data[rating_edge_name].edge_label_index
 
        if use_raw:
            df_user = pd.DataFrame(graph_data['user'].x[edge_label_index[0]].numpy(), columns=graph_data['user'].x_names)
            df_item = pd.DataFrame(graph_data[item_name].x[edge_label_index[1]].numpy(), columns=graph_data[item_name].x_names)
            df = pd.concat([df_user, df_item], axis=1)
        else:
            df = pd.DataFrame(graph_data[config.data.rating_edge_name].edge_label_predefined.numpy(), columns=graph_data[config.data.rating_edge_name].edge_label_predefined_names)
            df = df > 0.5  # Convert fuzzy values to pure boolean values
        df['ratings'] = graph_data[config.data.rating_edge_name].edge_label >= config.data.good_rating_threshold

        type_map = {
            np.dtype('int64'): 'NUMERIC',
            np.dtype('float64'): 'NUMERIC',
            np.dtype('bool'): ['False', 'True'],
            np.dtype('O'): 'STRING'
        }

        assert df.columns[-1] == 'ratings', "last column must be the ratings"

        arff_content = f'@relation {relation_name}\n'

        # Replace whitespaces in column names with underscores, and remove all non-alphanumeric characters
        df.columns = [col.replace(' ', '_') for col in df.columns]
        df.columns = [''.join(c for c in col if c.isalnum() or c == '_') for col in df.columns]

        for column in df.columns:
            if df[column].dtype in type_map:
                attr_type = type_map[df[column].dtype]
                if isinstance(attr_type, list):
                    attr_type = '{' + ','.join(attr_type) + '}'
                arff_content += f'@attribute {column} {attr_type}\n'
            else:
                unique_values = df[column].unique()
                attr_type = '{' + ','.join(map(str, unique_values)) + '}'
                arff_content += f'@attribute {column} {attr_type}\n'

        arff_content += '\n@data\n'
        arff_content += df.to_csv(index=False, header=False)

        with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_arff_file:
            temp_arff_file.write(arff_content)
            temp_arff_file.flush()  # Ensure data is written to the file
            loader = Loader(classname="weka.core.converters.ArffLoader")
            arff_data = loader.load_file(temp_arff_file.name)
            arff_data.class_is_last()

        return arff_data

    if not jvm_started:
        # Start the JVm
        jvm.start(max_heap_size="24g")
        jvm_started = True

    train_data_arff = graph_data_to_arff(config, train_data, use_raw)
    
    results_at_k = dict()
    for i in range(n_repeat_exp):
        # training
        jrip_baseline = Classifier(classname="weka.classifiers.rules.JRip")
        jrip_baseline.build_classifier(train_data_arff)
        logging.info(f'jrip_baseline={jrip_baseline}')

        # Store the classifier
        joblib.dump(jrip_baseline, os.path.join(config.results_path, f"jrip_classifier_run_{i}.joblib"))

        def model(graph_data: HeteroData):
            data_arff = graph_data_to_arff(config, graph_data, use_raw)
            predictions = []
            for index, inst in enumerate(data_arff):
                pred = jrip_baseline.classify_instance(inst)
                predictions.append(pred)
            return torch.Tensor(predictions)

        # compute scores for all user-item pairs to get top-k results
        results_at_k_this_run = evaluation_at_k(config, model(evaluation_data), evaluation_data[config.data.rating_edge_name].edge_label, evaluation_data['user'].num_nodes, evaluation_data[config.data.rating_edge_name].edge_label_index)
        logging.info(f"{results_at_k_this_run=}")
        if i == 0:
            for key in results_at_k_this_run:
                results_at_k[key] = [results_at_k_this_run[key]]
        else:
            for key in results_at_k_this_run:
                results_at_k[key].append(results_at_k_this_run[key])


    # Stop the JVM
    # TODO: move this all to an extra file
    # jvm.stop()

    return {
        key: results_at_k[key]
        for key in results_at_k
    }

@logging_decorator("Evaluate approach and baselines")
def evaluation(config: Config, 
               n_repeat_exp, 
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
    
    approach_results = test_approach(config, n_repeat_exp, device, train_data, evaluation_data, with_neural_network, num_of_learned_atoms, num_of_horn_clauses, learning_rates, num_of_epochs, batch_size, loss_fn, l1_lambda)

    results_dict = {
        'Our Model': approach_results
    }

    BASELINES = {
        # 'All Ones': (test_baseline_all_ones(config, n_repeat_exp, train_data, evaluation_data)),
        # 'All Zeros': (test_baseline_all_zeros(config, n_repeat_exp, train_data, evaluation_data)),
        'Decision Tree Orig': (test_baseline_decision_tree(config, n_repeat_exp, train_data, evaluation_data, use_raw=True)),
        'Decision Tree Pred': (test_baseline_decision_tree(config, n_repeat_exp, train_data, evaluation_data, use_raw=False)),
        # 'Random Forest Raw': (test_baseline_random_forest(config, n_repeat_exp, train_data, evaluation_data, use_raw=True)),
        'Surprise': (test_baseline_surprise(config, n_repeat_exp, train_data, evaluation_data)),
        # 'Random Forest Predefined': (test_baseline_random_forest(config, n_repeat_exp, train_data, evaluation_data, use_raw=False)),
        'LightGCN': (test_baseline_light_gcn(config, n_repeat_exp, train_data, evaluation_data, device, use_raw=True)),
        'JRip': (test_baseline_jrip(config, n_repeat_exp, train_data, evaluation_data, use_raw=False)),
    }

    for baseline_name, baseline_results in BASELINES.items():
        if baseline_name == 'Surprise':
            results_dict.update(baseline_results)
        else:
            results_dict[baseline_name] = baseline_results

    results_df = pd.DataFrame({
        _approach_: {_metric_: _values_ for _metric_, _values_ in _results_.items()}
        for _approach_, _results_ in results_dict.items()
    }).transpose()
    results_df.to_csv(os.path.join(config.results_path, "results.csv"))
