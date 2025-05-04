from torch_geometric.data import HeteroData
import pandas as pd
import torch
import numpy as np
import os
from diff_fnn.data.train_split import train_test_split
from diff_fnn.utils import logging_decorator, Config
import logging

HIGH_AVERAGE_RATING_THRESHOLD = np.array([4.0])
HIGH_AVERAGE_RATING_THRESHOLD_PER_USER = np.array([4.0])
MOVIE_OFTEN_RATED_THRESHOLD = np.array([228.])

def get_movielens_1m_graph(path):
    movie_path = os.path.join(path, 'movies.dat')
    user_path = os.path.join(path, 'users.dat')
    ratings_path = os.path.join(path, 'ratings.dat')
    movie_df = pd.read_csv(movie_path, sep="::", header=None, names=['MovieID', 'Title', 'Genres'], engine='python', encoding="iso-8859-1")
    user_df = pd.read_csv(user_path, sep="::", header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding="iso-8859-1")
    ratings_df = pd.read_csv(ratings_path, sep="::", header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding="iso-8859-1")

    movie_df['Year'] = movie_df['Title'].map(lambda title: int(title[:-1].rsplit('(', 1)[1]))
    movie_df['Title'] = movie_df['Title'].map(lambda title: title.rsplit('(', 1)[0].rstrip())
    x_movie_df = movie_df[['Year']]
    x_movie_df = pd.concat([x_movie_df, movie_df['Genres'].str.get_dummies(sep='|').add_prefix('Genre ')], axis=1)
    x_movie = torch.tensor(x_movie_df.to_numpy(), dtype=torch.float32)

    x_user_df = user_df[['Age']]
    x_user_df = pd.concat([x_user_df, pd.get_dummies(user_df['Gender'], prefix='Gender', dtype=int)], axis=1)
    x_user_df = pd.concat([x_user_df, pd.get_dummies(user_df['Occupation'], prefix='Occupation', dtype=int)], axis=1)
    x_user = torch.tensor(x_user_df.to_numpy(), dtype=torch.float32)

    # NOTE: movie_id does not start with 0, and also for the user
    edges_user_ids = ratings_df['UserID'].map(lambda original_id: np.where(user_df['UserID'].to_numpy() == original_id)[0][0])
    edges_movie_ids = ratings_df['MovieID'].map(lambda original_id: np.where(movie_df['MovieID'].to_numpy() == original_id)[0][0])
    edge_index = torch.tensor([edges_user_ids, edges_movie_ids])
    timestamps = torch.tensor(ratings_df['Timestamp'])
    ratings = torch.tensor(ratings_df['Rating'])

    data = HeteroData()
    data['movie'].x = x_movie
    data['movie'].x_names = list(x_movie_df.columns)
    data['user'].x = x_user
    data['user'].x_names = list(x_user_df.columns)
    data['user', 'rates', 'movie'].edge_index = edge_index
    data['user', 'rates', 'movie'].time = timestamps
    data['user', 'rates', 'movie'].edge_label = ratings
    return data

# define predefined atoms:
# NOTE: be cautious here to avoid data leakage, only use training data to get predefined atoms
def predefined_atoms_movielens1m(train_data: HeteroData, edge_index):
    user_edge_index = edge_index[0]
    movie_edge_index = edge_index[1]
    train_user_index = train_data['user', 'rates', 'movie'].edge_index[0]
    train_movie_index = train_data['user', 'rates', 'movie'].edge_index[1]

    # gender
    gender_names = ['Gender_F', 'Gender_M']
    gender_indices = [train_data['user'].x_names.index(n) for n in gender_names]
    gender = train_data['user'].x[:, gender_indices]

    # occupation
    occ_names = [f'Occupation_{i}' for i in range(21)]
    occ_indices = [train_data['user'].x_names.index(n) for n in occ_names]
    occupation = train_data['user'].x[:, occ_indices]

    # age
    age_index = train_data['user'].x_names.index('Age')
    age_numerical = train_data['user'].x[:, age_index].int()
    age_df = pd.get_dummies(age_numerical, prefix='Age')
    age_names = list(age_df.columns)
    age = torch.tensor(age_df.to_numpy()).float()

    # genre
    genre_names = [n for n in train_data['movie'].x_names if n.startswith('Genre ')]
    genre_indices = [train_data['movie'].x_names.index(n) for n in genre_names]
    genres = train_data['movie'].x[:, genre_indices]

    # year
    year_index = train_data['movie'].x_names.index('Year')
    year_numerical = train_data['movie'].x[:, year_index].int()
    year_numerical = (year_numerical // 10) * 10  # divide into decades
    year_df = pd.get_dummies(year_numerical, prefix='Dec')
    year_names = list(year_df.columns)
    year = torch.tensor(year_df.to_numpy()).float()

    # average rating of movie
    sum_ratings = torch.bincount(train_data['user', 'rates', 'movie'].edge_index[1], weights=train_data['user', 'rates', 'movie'].edge_label)
    count_ratings = torch.bincount(train_data['user', 'rates', 'movie'].edge_index[1])
    avg_ratings = sum_ratings / count_ratings.clamp(min=1)  # Avoid division by zero

    # percentage of how often does the user rated the genre (normalised with most often rated genre)
    rated_genres = genres[train_movie_index]
    # Count how often each user rated each genre
    user_genre_counts = torch.zeros((train_data['user'].num_nodes, genres.shape[1]))
    # Aggregate counts per user
    user_genre_counts.index_add_(0, train_user_index, rated_genres)
    # Normalize to get percentage (avoid division by zero)
    user_total_ratings = user_genre_counts.max(dim=1, keepdim=True).values
    user_rated_genre_percentage = user_genre_counts / user_total_ratings.clamp(min=1)  # Shape: [num_users, num_genres]

    # movie genre and rated genre dot product
    # COMBAK: use here or function
    genre_match = torch.sum(genres[movie_edge_index] * user_rated_genre_percentage[user_edge_index], dim=-1).clamp(max=1.0)

    # average rating of user
    sum_ratings_per_user = torch.bincount(train_user_index, weights=train_data['user', 'rates', 'movie'].edge_label)
    count_ratings_per_user = torch.bincount(train_user_index)
    avg_rating_per_user = sum_ratings_per_user / count_ratings_per_user.clamp(min=1)


    # correction for new users
    non_zero_entries = avg_rating_per_user[avg_rating_per_user != 0]
    mean_non_zero = non_zero_entries.mean().item()
    avg_rating_per_user[avg_rating_per_user == 0] = mean_non_zero
    # correction for new movies
    non_zero_entries = avg_ratings[avg_ratings != 0]
    mean_non_zero = non_zero_entries.mean().item()
    avg_ratings[avg_ratings == 0] = mean_non_zero


    predefined_atoms = gender[user_edge_index]
    predefined_atoms = torch.hstack((predefined_atoms, occupation[user_edge_index]))
    predefined_atoms = torch.hstack((predefined_atoms, age[user_edge_index]))
    predefined_atoms = torch.hstack((predefined_atoms, genres[movie_edge_index]))
    predefined_atoms = torch.hstack((predefined_atoms, year[movie_edge_index]))
    predefined_atoms = torch.hstack((predefined_atoms, user_rated_genre_percentage[user_edge_index]))
    predefined_atoms = torch.hstack((predefined_atoms, genre_match.unsqueeze(1)))
    for t in HIGH_AVERAGE_RATING_THRESHOLD:
        high_avg_ratings = (avg_ratings >= t).float()
        predefined_atoms = torch.hstack((predefined_atoms, high_avg_ratings[movie_edge_index].unsqueeze(1)))  # unsqueeze, since it is only one atom
    for t in HIGH_AVERAGE_RATING_THRESHOLD_PER_USER:
        high_avg_rating_per_user = (avg_rating_per_user >= t).float()
        predefined_atoms = torch.hstack((predefined_atoms, high_avg_rating_per_user[user_edge_index].unsqueeze(1)))
    for t in MOVIE_OFTEN_RATED_THRESHOLD:
        movie_rated_often = (count_ratings >= t).float()
        predefined_atoms = torch.hstack((predefined_atoms, movie_rated_often[movie_edge_index].unsqueeze(1)))

    names = gender_names
    names.extend(occ_names)
    names.extend(age_names)
    names.extend(genre_names)
    names.extend(year_names)
    names.extend(['User rated genre percentage ' + n for n in genre_names])
    names.extend(['Genre match'])
    for t in HIGH_AVERAGE_RATING_THRESHOLD:
        names.extend([f'High avg movie rating ({t}+)'])
    for t in HIGH_AVERAGE_RATING_THRESHOLD_PER_USER:
        names.extend([f'High avg rating per user ({t}+)'])
    for t in MOVIE_OFTEN_RATED_THRESHOLD:
        names.extend([f'Movie rated often ({t}+)'])
    names = np.array(names)

    return predefined_atoms, names

@logging_decorator("Preprocess data")
def preprocessing(config: Config):
    movielens_1m_graph = get_movielens_1m_graph("data/movielens1m/raw")
    movielens_1m_graph.validate()
    logging.info(f'{movielens_1m_graph=}')
    train_data, val_data, train_plus_val_data, test_data = train_test_split(config, movielens_1m_graph)
    # get predefined atoms
    train_data['user', 'rates', 'movie'].edge_label_predefined, train_data['user', 'rates', 'movie'].edge_label_predefined_names = predefined_atoms_movielens1m(
        train_data, 
        train_data['user', 'rates', 'movie'].edge_label_index
    )
    val_data['user', 'rates', 'movie'].edge_label_predefined, val_data['user', 'rates', 'movie'].edge_label_predefined_names = predefined_atoms_movielens1m(
        train_data, 
        val_data['user', 'rates', 'movie'].edge_label_index
    )
    train_plus_val_data['user', 'rates', 'movie'].edge_label_predefined, train_plus_val_data['user', 'rates', 'movie'].edge_label_predefined_names = predefined_atoms_movielens1m(
        train_plus_val_data, 
        train_plus_val_data['user', 'rates', 'movie'].edge_label_index
    )
    # NOTE: use train_plus_val_data for test_data
    test_data['user', 'rates', 'movie'].edge_label_predefined, test_data['user', 'rates', 'movie'].edge_label_predefined_names = predefined_atoms_movielens1m(
        train_plus_val_data, 
        test_data['user', 'rates', 'movie'].edge_label_index
    )
    return train_data, val_data, train_plus_val_data, test_data
