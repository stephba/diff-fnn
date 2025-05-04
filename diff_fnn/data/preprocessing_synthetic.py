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
    movie_is_recent = (np.random.rand(num_of_movies) < recent_prob)
    user_fav_genre = np.random.randint(0, num_of_genres, size=num_of_users)
    user_fav_actor = np.random.choice(range(num_of_actors), size=num_of_users, p=[0.205, 0.795])
    user_fav_director = np.random.choice(range(num_of_directors), size=num_of_users, p=[0.205, 0.795])
    movie_genre = np.random.randint(0, num_of_genres, size=num_of_movies)
    movie_actor = np.random.choice(range(num_of_actors), size=num_of_movies, p=[0.205, 0.795])
    movie_director = np.random.choice(range(num_of_directors), size=num_of_movies, p=[0.205, 0.795])
    same_genre = (user_fav_genre[user_ids] == movie_genre[movie_ids])
    same_actor = (user_fav_actor[user_ids] == movie_actor[movie_ids])
    same_director = (user_fav_director[user_ids] == movie_director[movie_ids])
    movie_high_rating = (np.random.rand(num_of_movies) < high_prob)
    # add useless features
    user_likes_cookies = (np.random.rand(num_of_users) < 0.5)

    ratings = np.where(movie_is_recent[movie_ids] & same_genre, True, False)
    ratings = np.logical_or(ratings, np.where(movie_high_rating[movie_ids], True, False))
    ratings = np.logical_or(ratings, np.where(movie_is_recent[movie_ids] & same_actor & same_director, True, False))

    timestamps = np.random.randint(1e6, 1e9, size=dataset_size)
    
    synthetic_data_df = pd.DataFrame(data={
        'user_id': user_ids,
        'movie_id': movie_ids,
        'timestamps': timestamps,
        'ratings': ratings
    })

    # remove duplicate ratings
    synthetic_data_df = synthetic_data_df.drop_duplicates(subset=['user_id', 'movie_id'])
    # only get subset to get a specific number of entries
    synthetic_data_df = synthetic_data_df.head(final_dataset_size)

    data = HeteroData()
    data['movie'].x = torch.tensor(np.array([
        movie_genre,
        movie_actor,
        movie_director,
        movie_is_recent,
        movie_high_rating,
    ])).T
    data['movie'].x_names = ['movie_genre', 'movie_actor', 'movie_director', 'movie_is_recent', 'movie_high_rating']
    data['user'].x = torch.tensor(np.array([
        user_fav_genre,
        user_fav_actor,
        user_fav_director,
        user_likes_cookies
    ])).T
    data['user'].x_names = ['user_fav_genre', 'user_fav_actor', 'user_fav_director', 'user_likes_cookies']
    data['user', 'rates', 'movie'].edge_index = torch.tensor(np.array([synthetic_data_df['user_id'].to_numpy(), synthetic_data_df['movie_id'].to_numpy()]))
    data['user', 'rates', 'movie'].time = torch.tensor(synthetic_data_df['timestamps'].to_numpy())
    data['user', 'rates', 'movie'].edge_label = torch.tensor(synthetic_data_df['ratings'].to_numpy()).float()
    return data

def get_predefined_atoms(train_data: HeteroData, edge_index):
    user_edge_index = edge_index[0]
    movie_edge_index = edge_index[1]

    recent_index = 3
    recent = train_data['movie'].x[:, recent_index]

    genre_index = 0
    same_genre = (train_data['user'].x[:, genre_index][user_edge_index] == train_data['movie'].x[:, genre_index][movie_edge_index])

    actor_index = 1
    same_actor = (train_data['user'].x[:, actor_index][user_edge_index] == train_data['movie'].x[:, actor_index][movie_edge_index])

    director_index = 2
    same_director = (train_data['user'].x[:, director_index][user_edge_index] == train_data['movie'].x[:, director_index][movie_edge_index])

    high_rating_index = 4
    high_rating = train_data['movie'].x[:, high_rating_index]

    cookies_index = 3
    cookies = train_data['user'].x[:, cookies_index]

    predefined_atoms = recent[movie_edge_index].unsqueeze(1)
    predefined_atoms = torch.hstack((predefined_atoms, same_genre.unsqueeze(1)))
    predefined_atoms = torch.hstack((predefined_atoms, same_actor.unsqueeze(1)))
    predefined_atoms = torch.hstack((predefined_atoms, same_director.unsqueeze(1)))
    predefined_atoms = torch.hstack((predefined_atoms, high_rating[movie_edge_index].unsqueeze(1)))
    predefined_atoms = torch.hstack((predefined_atoms, cookies[user_edge_index].unsqueeze(1)))

    names = np.array(['movie is recent', "movie genre is user's favourite genre", "same actor", "same director", 'movie has high rating', 'user likes cookies'])

    return predefined_atoms, names

@logging_decorator("Preprocess data")
def preprocessing(config: Config):
    synthetic_graph = generate_synthetic_graph()
    synthetic_graph.validate()
    logging.info(f'{synthetic_graph=}')
    train_data, val_data, train_plus_val_data, test_data = train_test_split(config, synthetic_graph)
    # get predefined atoms
    train_data['user', 'rates', 'movie'].edge_label_predefined, train_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_graph, 
        train_data['user', 'rates', 'movie'].edge_label_index
    )
    val_data['user', 'rates', 'movie'].edge_label_predefined, val_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_graph, 
        val_data['user', 'rates', 'movie'].edge_label_index
    )
    train_plus_val_data['user', 'rates', 'movie'].edge_label_predefined, train_plus_val_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_graph, 
        train_plus_val_data['user', 'rates', 'movie'].edge_label_index
    )
    test_data['user', 'rates', 'movie'].edge_label_predefined, test_data['user', 'rates', 'movie'].edge_label_predefined_names = get_predefined_atoms(
        synthetic_graph, 
        test_data['user', 'rates', 'movie'].edge_label_index
    )
    return train_data, val_data, train_plus_val_data, test_data
