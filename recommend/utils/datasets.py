import os
from six.moves import xrange
from functools import partial
import numpy as np
from numpy.random import RandomState
import scipy.sparse as sparse


def make_ratings(n_users, n_items, min_rating_per_user, max_rating_per_user,
                 rating_choices, seed=None, shuffle=True):
    """Randomly generate a (user_id, item_id, rating) array

    Return
    ------
        ndarray with shape (n_samples, 3)

    """
    if not (isinstance(rating_choices, list) or
            isinstance(rating_choices, tuple)):
        raise ValueError("'rating_choices' must be a list or tuple")
    if min_rating_per_user < 0 or min_rating_per_user >= n_items:
        raise ValueError("invalid 'min_rating_per_user' invalid")
    if (min_rating_per_user > max_rating_per_user) or \
       (max_rating_per_user >= n_items):
        raise ValueError("invalid 'max_rating_per_user' invalid")

    rs = RandomState(seed=seed)
    user_arrs = []
    for user_id in xrange(n_users):
        item_count = rs.randint(min_rating_per_user, max_rating_per_user)
        item_ids = rs.choice(n_items, item_count, replace=False)
        ratings = rs.choice(rating_choices, item_count)
        arr = np.stack(
            [np.repeat(user_id, item_count), item_ids, ratings], axis=1)
        user_arrs.append(arr)

    ratings = np.array(np.vstack(user_arrs))
    ratings[:, 2] = ratings[:, 2].astype('float')
    if shuffle:
        rs.shuffle(ratings)
    return ratings


def load_movielens_ratings(ratings_file, separator):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
            line = line.split(separator)[:3]
            line = [int(l) for l in line]
            ratings.append(line)
        ratings = np.array(ratings)
    return ratings


def load_netflix_ratings(ratings_file, separator):
    with open(ratings_file) as f:
        ratings = []
        for line in f:
            line = line.split(separator)
            line = [line[0], line[1], line[3]] # ignore date
            line = [int(l) for l in line]
            if line[2] != 0:
                ratings.append(line)
        ratings = np.array(ratings)
    return ratings


def load_netflix(filename):
    if not os.path.isfile(filename + ".npy"):
        print("Loading fresh")
        separator = " "
        ratings = load_netflix_ratings(filename + ".dta", separator)
        print("Saving")
        ratings = [item for sublist in ratings for item in sublist]
        np.save(filename, ratings)
    ratings = np.load(filename + ".npy").tolist()
    num_ratings = len(ratings)
    if (num_ratings % 3 != 0):
        raise ValueError("Loaded file not factor of 3")
    ratings = np.reshape(ratings, (num_ratings / 3, 3))
    # shift user_id & movie_id by 1. let user_id & movie_id start from 0
    ratings[:, (0, 1)] -= 1
    print("Finished loading")
    return ratings

load_movielens_1m_ratings = partial(load_movielens_ratings, separator="::")
load_movielens_100k_ratings = partial(load_movielens_ratings, separator="\t")


def build_user_item_matrix(n_users, n_items, ratings):
    """Build user-item matrix

    Return
    ------
        sparse matrix with shape (n_users, n_items)
    """
    data = ratings[:, 2]
    row_ind = ratings[:, 0]
    col_ind = ratings[:, 1]
    shape = (n_users, n_items)
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
