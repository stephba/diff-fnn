import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
from torch_geometric.data import HeteroData
torch.set_printoptions(sci_mode=False)
from diff_fnn.utils import Config
from abc import ABC, abstractmethod
from collections import namedtuple

### t norm ###
class FuzzyOperators(ABC):
    @abstractmethod
    def _not_(x):
        raise NotImplementedError
    @abstractmethod
    def _and_(x, y):
        raise NotImplementedError
    @abstractmethod
    def _or_(x, y):
        raise NotImplementedError
    @abstractmethod
    def _or_multiple_(variables):
        raise NotImplementedError
    @abstractmethod
    def _and_multiple_(variables):
        raise NotImplementedError

class ProductTNormOperators(FuzzyOperators):
    def _not_(x):
        return 1 - x
    def _and_(x, y):
        return x * y
    def _or_(x, y):
        return x + y - x * y
    def _or_multiple_(variables):
        """
        Apply OR recursively, to create OR with more than 2 inputs
        """
        num_of_variables = variables.shape[-1]
        if num_of_variables < 2:
            return variables.squeeze()  # remove one dimension
        # OR with the first two variables
        result = ProductTNormOperators._or_(variables[:,0], variables[:,1])
        # OR with the rest of the variables
        for var_idx in range(2, num_of_variables):
            result = ProductTNormOperators._or_(result,variables[:, var_idx])
        return result
    def _and_multiple_(variables):
        """
        Apply product t norm and on multiple variables
        """
        return torch.prod(variables, dim=1)

### GNN ###

### Neural Network ###

class HornLayer(nn.Module):
    def __init__(self, 
                 fuzzy_operators: FuzzyOperators, 
                 body_size):
        super().__init__()
        self.ops = fuzzy_operators
        self.body_size = body_size
        weights = torch.Tensor(body_size)
        self.weights = nn.Parameter(weights)

        # initialise weights
        nn.init.normal_(self.weights, mean=0.0, std=1.0)

    def forward(self, body, extraction_threshold):
        # apply sigmoid to weights to get values between 0 and 1
        fuzzy_weights = self.get_fuzzy_weights(extraction_threshold)
        y = self.ops._or_(body, 1 - fuzzy_weights)
        y = self.ops._and_multiple_(y)
        return y

    def get_fuzzy_weights(self, extraction_threshold):
        fuzzy_weights = torch.sigmoid(self.weights)
        if extraction_threshold:
            fuzzy_weights = torch.where(fuzzy_weights < extraction_threshold, torch.tensor(0.0), fuzzy_weights)
        return fuzzy_weights


HornNetworkArgs = namedtuple('HornNetworkArgs', [
    'fuzzy_operators', 'config', 'predefined_input_size',
    'num_of_horn_clauses', 'num_learned_atoms', 'with_neural_net'
])

class HornNetwork(nn.Module):
    def __init__(self, args: HornNetworkArgs):
        super().__init__()
        self.ops = args.fuzzy_operators
        self.config = args.config
        self.predefined_input_size = args.predefined_input_size
        self.num_learned_atoms = args.num_learned_atoms
        self.num_of_horn_clauses = args.num_of_horn_clauses
        self.with_neural_net = args.with_neural_net

        # body contains learned as well as predefined atoms
        if self.with_neural_net:
            self.body_size = self.num_learned_atoms + self.predefined_input_size
        else:
            self.body_size = self.predefined_input_size
        self.horn_layers = nn.ModuleList([
            HornLayer(self.ops, self.body_size) for _ in range(self.num_of_horn_clauses)
        ])
        if self.with_neural_net:
            raise NotImplementedError
            self.neural_net = MultiClassClassifier(raw_input_size, num_learned_atoms)

    def get_constructor_arguments(self):
        return HornNetworkArgs(
            fuzzy_operators=self.ops,
            config=self.config,
            predefined_input_size=self.predefined_input_size,
            num_of_horn_clauses=self.num_of_horn_clauses,
            num_learned_atoms=self.num_learned_atoms,
            with_neural_net=self.with_neural_net
        )

    def forward(self, 
                graph_data: HeteroData, extraction_threshold=None):
        outputs = []

        edge_name = self.config.data.rating_edge_name
        predefined_atoms = graph_data[edge_name].edge_label_predefined

        # NOTE: multiclass classification for learned atoms
        if self.with_neural_net:
            raise NotImplementedError
            learned_atoms = self.neural_net(raw_input)
            body = torch.cat((learned_atoms, predefined_atoms), dim=1)
        else:
            body = predefined_atoms

        for horn_layer in self.horn_layers:
            outputs.append(horn_layer(body, extraction_threshold))
        all_layer_outputs = torch.stack(outputs, dim=0).T
        return self.ops._or_multiple_(all_layer_outputs)
    
    def get_fuzzy_weights(self, extraction_threshold=None):
        all_fuzzy_weights = []
        for horn_layer in self.horn_layers:
            all_fuzzy_weights.append(horn_layer.get_fuzzy_weights(extraction_threshold))
        all_fuzzy_weights = torch.stack(all_fuzzy_weights, dim=0)
        return all_fuzzy_weights
    
    def fit(model, 
            train_data: HeteroData, 
            evaluation_data: HeteroData, 
            learning_rates, loss_fn, l1_lambda, num_of_epochs):
        param_group = [
            {'params': model.horn_layers.parameters(), 'lr': learning_rates['horn_layers']}
        ]
        if model.with_neural_net:
            param_group.append({'params': model.neural_net.parameters(), 'lr': learning_rates['neural_net']})
        optimizer = torch.optim.Adam(param_group)

        # NOTE: more values can be nonzero if we want to get more clauses
        l1_lambda = l1_lambda / (model.num_of_horn_clauses * (model.predefined_input_size + model.num_learned_atoms))

        edge_name = model.config.data.rating_edge_name

        train_losses = []
        val_losses = []
        for epoch in tqdm(range(num_of_epochs)):            
            optimizer.zero_grad()

            # Training phase
            model.train()
            y_pred = model(train_data)
            target = (train_data[edge_name].edge_label >= model.config.data.good_rating_threshold).float()
            loss = loss_fn(y_pred, target)

            # add l1 regularisation for each variable -> small rules
            fuzzy_weights = model.get_fuzzy_weights()
            loss += l1_lambda * torch.norm(fuzzy_weights, p=1)

            # backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Validation phase
            model.eval()
            with torch.no_grad():
                y_val_pred = model(evaluation_data)
                val_target = (evaluation_data[edge_name].edge_label >= model.config.data.good_rating_threshold).float()
                val_loss = loss_fn(y_val_pred, val_target)
                val_losses.append(val_loss.item())
        return train_losses, val_losses
    
def get_data_with_pruned_atoms(graph_data: HeteroData, rating_edge_name, learned_fuzzy_weights, pruning_threshold):
    pruned_atoms_mask = torch.all(learned_fuzzy_weights < pruning_threshold, dim=0)
    pruned_data = graph_data.clone()
    predefined_names = pruned_data[rating_edge_name]['edge_label_predefined_names']
    pruned_data[rating_edge_name]['edge_label_predefined_names'] = predefined_names[~pruned_atoms_mask]
    predefined_atoms = pruned_data[rating_edge_name]['edge_label_predefined']
    pruned_data[rating_edge_name]['edge_label_predefined'] = predefined_atoms[:, ~pruned_atoms_mask]
    return pruned_data, pruned_atoms_mask

def get_pruned_model(old_model: HornNetwork, train_data: HeteroData, evaluation_data: HeteroData, device, rating_edge_name, pruning_threshold, learning_rates, loss_fn, l1_lambda, num_of_epochs):
    train_data_pruned, pruned_atoms_mask = get_data_with_pruned_atoms(train_data, rating_edge_name, old_model.get_fuzzy_weights(), pruning_threshold)
    evaluation_data_pruned, _ = get_data_with_pruned_atoms(evaluation_data, rating_edge_name, old_model.get_fuzzy_weights(), pruning_threshold)
    num_of_atoms = torch.sum(~pruned_atoms_mask).item()
    network_args = old_model.get_constructor_arguments()
    pruned_network_args = HornNetworkArgs(
        fuzzy_operators=network_args.fuzzy_operators,
        config=network_args.config,
        predefined_input_size=num_of_atoms, # use correct number of atoms (due to pruning)
        num_of_horn_clauses=network_args.num_of_horn_clauses,
        num_learned_atoms=network_args.num_learned_atoms,
        with_neural_net=network_args.with_neural_net
    )
    pruned_model = HornNetwork(pruned_network_args).to(device)
    train_losses, val_losses = pruned_model.fit(train_data_pruned, evaluation_data_pruned, learning_rates, loss_fn, l1_lambda, num_of_epochs)

    return pruned_model, train_losses, val_losses
