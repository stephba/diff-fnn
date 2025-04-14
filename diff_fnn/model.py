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

### t norm ###

def prod_not(x):
    return 1 - x

def prod_and(x, y):
    return x * y

def prod_or(x, y):
    return x + y - x * y

def prod_or_multiple(variables):
    """
    Apply OR recursively, to create OR with more than 2 inputs
    """
    num_of_variables = variables.shape[-1]
    if num_of_variables < 2:
        return variables.squeeze()  # remove one dimension
    # OR with the first two variables
    result = prod_or(variables[:,0], variables[:,1])
    # OR with the rest of the variables
    for var_idx in range(2, num_of_variables):
        result = prod_or(result,variables[:, var_idx])
    return result

def prod_and_multiple(variables):
    """
    Apply product t norm and on multiple variables
    """
    return torch.prod(variables, dim=1)

### GNN ###

### Neural Network ###

class HornLayer(nn.Module):
    # NOTE: these operators are only implemented using product t norm
    def __not(self, x):
        return prod_not(x)
    def __and(self, x, y):
        return prod_and(x, y)
    def __or(self, x, y):
        return prod_or(x, y)
    def __or_multiple(self, variables):
        return prod_or_multiple(variables)
    def __and_multiple(self, variables):
        return prod_and_multiple(variables)

    def __init__(self, body_size, extraction_threshold=0.1):
        super().__init__()
        self.body_size = body_size
        self.extraction_threshold = extraction_threshold
        weights = torch.Tensor(body_size)
        self.weights = nn.Parameter(weights)

        # initialise weights
        nn.init.normal_(self.weights, mean=0.0, std=1.0)

    def forward(self, body, extract_rules):
        # apply sigmoid to weights to get values between 0 and 1
        fuzzy_weights = self.get_fuzzy_weights(extract_rules)
        y = self.__or(body, 1 - fuzzy_weights)
        y = self.__and_multiple(y)
        return y

    def get_fuzzy_weights(self, extract_rules):
        fuzzy_weights = torch.sigmoid(self.weights)
        if extract_rules:
            fuzzy_weights = torch.where(fuzzy_weights < self.extraction_threshold, torch.tensor(0.0), fuzzy_weights)
        return fuzzy_weights
    

class HornNetwork(nn.Module):
    def __init__(self, config: Config, predefined_input_size, num_of_horn_clauses, num_learned_atoms=0, with_neural_net=False):
        super().__init__()
        self.config = config
        self.predefined_input_size = predefined_input_size
        self.num_learned_atoms = num_learned_atoms
        self.num_of_horn_clauses = num_of_horn_clauses
        self.with_neural_net = with_neural_net
        # body contains learned as well as predefined atoms
        if self.with_neural_net:
            self.body_size = self.num_learned_atoms + self.predefined_input_size
        else:
            self.body_size = self.predefined_input_size
        self.horn_layers = nn.ModuleList([
            HornLayer(self.body_size) for _ in range(num_of_horn_clauses)
        ])
        if self.with_neural_net:
            raise NotImplementedError
            self.neural_net = MultiClassClassifier(raw_input_size, num_learned_atoms)

    def forward(self, 
                graph_data: HeteroData, extract_rules=False):
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
            outputs.append(horn_layer(body, extract_rules))
        all_layer_outputs = torch.stack(outputs, dim=0).T
        return prod_or_multiple(all_layer_outputs)
    
    def get_fuzzy_weights(self, extract_rules=False):
        all_fuzzy_weights = []
        for horn_layer in self.horn_layers:
            all_fuzzy_weights.append(horn_layer.get_fuzzy_weights(extract_rules))
        all_fuzzy_weights = torch.stack(all_fuzzy_weights, dim=0)
        return all_fuzzy_weights
    
    def fit(model, 
            train_data: HeteroData, 
            evaluation_data: HeteroData, 
            optimizer, loss_fn, l1_lambda, num_of_epochs):
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
