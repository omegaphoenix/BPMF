from __future__ import print_function

import os
import logging
import zipfile
from six.moves import urllib
from numpy.random import RandomState
from recommend.bpmf import BPMF
from recommend.utils.evaluation import RMSE
# from recommend.utils.datasets import load_movielens_1m_ratings
from recommend.utils.datasets import load_netflix

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# ML_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
# ML_1M_FOLDER = "ml-1m"
# ML_1M_ZIP_SIZE = 24594131

rand_state = RandomState(0)

# download MovieLens 1M dataset if necessary
# def ml_1m_download(folder, file_size):
#     file_name = "ratings.dat"
#     file_path = os.path.join(os.getcwd(), folder, file_name)
#     if not os.path.exists(file_path):
#         print("file %s not exists. downloading..." % file_path)
#         zip_name, _ = urllib.request.urlretrieve(ML_1M_URL, "ml-1m.zip")
#         with zipfile.ZipFile(zip_name, 'r') as zf:
#             file_path = zf.extract('ml-1m/ratings.dat')
# 
#     # check file
#     statinfo = os.stat(file_path)
#     if statinfo.st_size == file_size:
#         print('verify success: %s' % file_path)
#     else:
#         raise Exception('verify failed: %s' % file_path)
#     return file_path

# load or download MovieLens 1M dataset
# rating_file = ml_1m_download(ML_1M_FOLDER, file_size=ML_1M_ZIP_SIZE)
train_filename = "/Users/justinleong/redeem-team/data/um/1236"
ratings = load_netflix(train_filename)
n_user = max(ratings[:, 0]) + 1
n_item = max(ratings[:, 1]) + 1

rand_state.shuffle(ratings)
print("Done shuffling\n")
train_size = ratings.shape[0]
train = ratings

test_filename = "/Users/justinleong/redeem-team/data/um/4"
ratings = load_netflix(test_filename)

# rand_state.shuffle(ratings)
test_size = ratings.shape[0]
validation = ratings

# models settings
n_feature = 10
eval_iters = 100
print("n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,
            max_rating=5., min_rating=1., seed=0)

bpmf.fit(train, n_iters=eval_iters)
train_preds = bpmf.predict(train[:, :2])
train_rmse = RMSE(train_preds, train[:, 2])
val_preds = bpmf.predict(validation[:, :2])
val_rmse = RMSE(val_preds, validation[:, 2])
print("after %d iteration, train RMSE: %.6f, validation RMSE: %.6f" %
      (eval_iters, train_rmse, val_rmse))
