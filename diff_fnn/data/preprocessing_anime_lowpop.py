from torch_geometric.data import HeteroData
import pandas as pd
import torch
import numpy as np
import os
from diff_fnn.data.train_split import train_test_split
from diff_fnn.utils import logging_decorator, Config
import logging

HIGH_AVERAGE_RATING_THRESHOLD = np.array([7.05714286])
HIGH_AVERAGE_RATING_THRESHOLD_PER_USER = np.array([7.1618378])
ITEM_OFTEN_RATED_THRESHOLD = np.array([92.])

def get_anime_graph(path):
    cats_df = pd.read_csv(os.path.join(path, 'categories.txt')).set_index('index')
    low_pop_users_df = pd.read_csv(os.path.join(path, 'low_main_users.txt'))
    ratings_df = pd.read_csv(os.path.join(path, 'user_events_cats.txt'))
    # NOTE: only load the lowpop users
    ratings_df = ratings_df.merge(low_pop_users_df, on='user', how='inner')

    item_df = ratings_df[['item', 'cats']].drop_duplicates()
    # replace indices with names
    item_df['cats'] = item_df['cats'].apply(lambda x: '|'.join(
        cats_df['cat'].reindex(map(int, x.split('|'))).astype(str)
        ))
    x_item_df = item_df['cats'].str.get_dummies(sep='|').add_prefix('Category ')
    x_item = torch.tensor(x_item_df.to_numpy(), dtype=torch.float32)

    user_df = ratings_df[['user']].drop_duplicates()
    num_of_users = len(user_df['user'])

    # NOTE: item id does not start with 0, and also for the user
    edges_user_ids = ratings_df['user'].map(lambda original_id: np.where(user_df['user'].to_numpy() == original_id)[0][0])
    edges_item_ids = ratings_df['item'].map(lambda original_id: np.where(item_df['item'].to_numpy() == original_id)[0][0])
    edge_index = torch.tensor([edges_user_ids, edges_item_ids])
    # No timestamps available
    timestamps = torch.tensor(np.random.randint(1e6, 1e9, size=len(ratings_df.index)))
    ratings = torch.tensor(ratings_df['preference'])

    data = HeteroData()
    data['item'].x = x_item
    data['item'].x_names = list(x_item_df.columns)
    data['user'].x = torch.ones((num_of_users, 1))  # no user information available
    data['user', 'rates', 'item'].edge_index = edge_index
    data['user', 'rates', 'item'].time = timestamps
    data['user', 'rates', 'item'].edge_label = ratings
    return data

def predefined_atoms_anime(train_data: HeteroData, edge_index):
    user_edge_index = edge_index[0]
    item_edge_index = edge_index[1]
    train_user_index = train_data['user', 'rates', 'item'].edge_index[0]
    train_item_index = train_data['user', 'rates', 'item'].edge_index[1]
    num_of_items = train_data['item'].x.shape[0]
    num_of_users = train_data['user'].x.shape[0]

    # categories
    cat_names = [n for n in train_data['item'].x_names if n.startswith('Category ')]
    cat_indices = [train_data['item'].x_names.index(n) for n in cat_names]
    cats = train_data['item'].x[:, cat_indices]

    # average rating of item
    sum_ratings = torch.bincount(train_data['user', 'rates', 'item'].edge_index[1], weights=train_data['user', 'rates', 'item'].edge_label, minlength=num_of_items)
    count_ratings = torch.bincount(train_data['user', 'rates', 'item'].edge_index[1], minlength=num_of_items)
    avg_ratings = sum_ratings / count_ratings.clamp(min=1)  # Avoid division by zero

    # percentage of how often does the user rated the category (normalised with most often rated genre)
    rated_cats = cats[train_item_index]
    user_genre_counts = torch.zeros((train_data['user'].num_nodes, cats.shape[1]))
    user_genre_counts.index_add_(0, train_user_index, rated_cats)
    user_total_ratings = user_genre_counts.max(dim=1, keepdim=True).values
    user_rated_genre_percentage = user_genre_counts / user_total_ratings.clamp(min=1)  # Shape: [num_users, num_genres]

    # item cat and rated cat dot product
    # COMBAK: use here or function
    cat_match = torch.sum(cats[item_edge_index] * user_rated_genre_percentage[user_edge_index], dim=-1).clamp(max=1.0)

    # average rating of user
    sum_ratings_per_user = torch.bincount(train_user_index, weights=train_data['user', 'rates', 'item'].edge_label, minlength=num_of_users)
    count_ratings_per_user = torch.bincount(train_user_index, minlength=num_of_users)
    avg_rating_per_user = sum_ratings_per_user / count_ratings_per_user.clamp(min=1)

    # correction for new users
    non_zero_entries = avg_rating_per_user[avg_rating_per_user != 0]
    mean_non_zero = non_zero_entries.mean().item()
    avg_rating_per_user[avg_rating_per_user == 0] = mean_non_zero
    # correction for new items
    non_zero_entries = avg_ratings[avg_ratings != 0]
    mean_non_zero = non_zero_entries.mean().item()
    avg_ratings[avg_ratings == 0] = mean_non_zero

    predefined_atoms = cats[item_edge_index]
    for t in HIGH_AVERAGE_RATING_THRESHOLD:
        high_avg_ratings = (avg_ratings >= t).float()
        predefined_atoms = torch.hstack((predefined_atoms, high_avg_ratings[item_edge_index].unsqueeze(1)))  # unsqueeze, since it is only one atom
    predefined_atoms = torch.hstack((predefined_atoms, user_rated_genre_percentage[user_edge_index]))
    predefined_atoms = torch.hstack((predefined_atoms, cat_match.unsqueeze(1)))
    for t in HIGH_AVERAGE_RATING_THRESHOLD_PER_USER:
        high_avg_rating_per_user = (avg_rating_per_user >= t).float()
        predefined_atoms = torch.hstack((predefined_atoms, high_avg_rating_per_user[user_edge_index].unsqueeze(1)))
    for t in ITEM_OFTEN_RATED_THRESHOLD:
        item_rated_often = (count_ratings >= t).float()
        predefined_atoms = torch.hstack((predefined_atoms, item_rated_often[item_edge_index].unsqueeze(1)))

    names = cat_names.copy()
    for t in HIGH_AVERAGE_RATING_THRESHOLD:
        names.extend([f'High avg item rating ({t}+)'])
    names.extend(['User rated genre percentage ' + n for n in cat_names])
    names.extend(['Category match'])
    for t in HIGH_AVERAGE_RATING_THRESHOLD_PER_USER:
        names.extend([f'High avg rating per user ({t}+)'])
    for t in ITEM_OFTEN_RATED_THRESHOLD:
        names.extend([f'Item rated often ({t}+)'])
    names = np.array(names)

    return predefined_atoms, names

@logging_decorator("Preprocess data")
def preprocessing(config: Config):
    anime_graph = get_anime_graph("data/anime/raw")
    anime_graph.validate()
    logging.info(f'{anime_graph=}')
    train_data, val_data, train_plus_val_data, test_data = train_test_split(config, anime_graph)
    # get predefined atoms
    train_data['user', 'rates', 'item'].edge_label_predefined, train_data['user', 'rates', 'item'].edge_label_predefined_names = predefined_atoms_anime(
        train_data, 
        train_data['user', 'rates', 'item'].edge_label_index
    )
    val_data['user', 'rates', 'item'].edge_label_predefined, val_data['user', 'rates', 'item'].edge_label_predefined_names = predefined_atoms_anime(
        train_data, 
        val_data['user', 'rates', 'item'].edge_label_index
    )
    train_plus_val_data['user', 'rates', 'item'].edge_label_predefined, train_plus_val_data['user', 'rates', 'item'].edge_label_predefined_names = predefined_atoms_anime(
        train_plus_val_data, 
        train_plus_val_data['user', 'rates', 'item'].edge_label_index
    )
    # NOTE: use train_plus_val_data for test_data
    test_data['user', 'rates', 'item'].edge_label_predefined, test_data['user', 'rates', 'item'].edge_label_predefined_names = predefined_atoms_anime(
        train_plus_val_data, 
        test_data['user', 'rates', 'item'].edge_label_index
    )
    return train_data, val_data, train_plus_val_data, test_data
