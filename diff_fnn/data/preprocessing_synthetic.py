from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch
from diff_fnn.data.train_split import train_test_split
from diff_fnn.utils import logging_decorator, Config
import logging

@logging_decorator("Generate synthetic data")
def generate_synthetic_graph():
    np.random.seed(42)

    dataset_size = 1500000
    final_dataset_size = 1000209
    num_of_users = 6040
    num_of_movies = 3883
    num_of_genres = 2
    num_of_actors = 2
    num_of_directors = 2

    # useful features
    # probabilities are based on the equation that all rules are equally important and the total probability sums up to 0.5
    high_prob = 0.206299474015900
    recent_prob = 0.4542020189

    user_ids = np.random.randint(0, num_of_users, size=dataset_size)
    movie_ids = np.random.randint(0, num_of_movies, size=dataset_size)
    movie_is_recent = (np.random.rand(num_of_movies) < recent_prob)[movie_ids]
    user_fav_genre = np.random.randint(0, num_of_genres, size=num_of_users)[user_ids]
    user_fav_actor = np.random.choice(range(num_of_actors), size=num_of_users, p=[0.205, 0.795])[user_ids]
    user_fav_director = np.random.choice(range(num_of_directors), size=num_of_users, p=[0.205, 0.795])[user_ids]
    movie_genre = np.random.randint(0, num_of_genres, size=num_of_movies)[movie_ids]
    movie_actor = np.random.choice(range(num_of_actors), size=num_of_movies, p=[0.205, 0.795])[movie_ids]
    movie_director = np.random.choice(range(num_of_directors), size=num_of_movies, p=[0.205, 0.795])[movie_ids]
    same_genre = (user_fav_genre == movie_genre)
    same_actor = (user_fav_actor == movie_actor)
    same_director = (user_fav_director == movie_director)
    movie_high_rating = (np.random.rand(num_of_movies) < high_prob)[movie_ids]
    # add useless features
    user_likes_cookies = (np.random.rand(num_of_users) < 0.5)[user_ids]

    ratings = np.where(movie_is_recent & same_genre, True, False)
    ratings = np.logical_or(ratings, np.where(movie_high_rating, True, False))
    ratings = np.logical_or(ratings, np.where(movie_is_recent & same_actor & same_director, True, False))

    timestamps = np.random.randint(1e6, 1e9, size=dataset_size)
    
    synthetic_data_df = pd.DataFrame(data={
        'user_id': user_ids,
        'movie_id': movie_ids,
        'movie is recent': movie_is_recent,
        "movie genre is user's favourite genre": same_genre,
        "same actor": same_actor,
        "same director": same_director,
        'movie has high rating': movie_high_rating,
        'user likes cookies': user_likes_cookies,
        'timestamps': timestamps,
        'ratings': ratings
    })

    # remove duplicate ratings
    synthetic_data_df = synthetic_data_df.drop_duplicates(subset=['user_id', 'movie_id'])
    # only get subset to get a specific number of entries
    synthetic_data_df = synthetic_data_df.head(final_dataset_size)

    data = HeteroData()
    data['movie'].x = torch.ones((num_of_movies, 1))
    data['user'].x = torch.ones((num_of_users, 1))
    data['user', 'rates', 'movie'].edge_index = torch.tensor(np.array([synthetic_data_df['user_id'].to_numpy(), synthetic_data_df['movie_id'].to_numpy()]))
    data['user', 'rates', 'movie'].time = torch.tensor(synthetic_data_df['timestamps'].to_numpy())
    data['user', 'rates', 'movie'].edge_label = torch.tensor(synthetic_data_df['ratings'].to_numpy()).float()
    return data, synthetic_data_df

def get_predefined_atoms(synthetic_data_df, edge_index):
    edge_df = pd.DataFrame({
        'user_id': edge_index[0],
        'movie_id': edge_index[1]
    })
    predefined_df = edge_df.merge(synthetic_data_df, on=['user_id', 'movie_id'], how='left')
    names = ['movie is recent', "movie genre is user's favourite genre", "same actor", "same director", 'movie has high rating', 'user likes cookies']
    predefined_atoms = torch.tensor(predefined_df[names].to_numpy())
    return predefined_atoms, names

@logging_decorator("Preprocess data")
def preprocessing(config: Config):
    synthetic_graph, synthetic_data_df = generate_synthetic_graph()
    synthetic_graph.validate()
    logging.info(f'{synthetic_graph=}')
    train_data, val_data, train_plus_val_data, test_data = train_test_split(config, synthetic_graph)
    # get predefined atoms
    train_data['user', 'rates', 'movie'].edge_label_predefined, train_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_data_df, 
        train_data['user', 'rates', 'movie'].edge_label_index
    )
    val_data['user', 'rates', 'movie'].edge_label_predefined, val_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_data_df, 
        val_data['user', 'rates', 'movie'].edge_label_index
    )
    train_plus_val_data['user', 'rates', 'movie'].edge_label_predefined, train_plus_val_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_data_df, 
        train_plus_val_data['user', 'rates', 'movie'].edge_label_index
    )
    test_data['user', 'rates', 'movie'].edge_label_predefined, test_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_data_df, 
        test_data['user', 'rates', 'movie'].edge_label_index
    )
    return train_data, val_data, train_plus_val_data, test_data
