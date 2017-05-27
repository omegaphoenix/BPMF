"""
Base class of recommendation System
"""

from abc import ABCMeta, abstractmethod

from .utils.validation import output_predictions
from .utils.evaluation import RMSE
from recommend.utils.datasets import load_netflix
import logging

logger = logging.getLogger(__name__)


class ModelBase(object):

    """base class of recommendations"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, train, n_iters):
        """training models"""

    @abstractmethod
    def predict(self, data):
        """save model"""

    def output_all(self, filename):
        """output all predictions"""
        in_filename = "/Users/justinleong/redeem-team/data/um/"
        for i in xrange(5):
            j = str(i + 1)
            cur_filename = in_filename + j
            if i == 4: # qual
                cur_filename = cur_filename + "-1"
            pred_ratings = load_netflix(cur_filename)
            preds = self.predict(pred_ratings[:, :2])
            if i == 3: # probe
                # compute RMSE
                test_rmse = RMSE(preds, pred_ratings[:, 2])
                logger.info("test RMSE: %.6f", test_rmse)
            out_filename = in_filename + "data_" + j + ".txt"
            output_predictions(preds, out_filename)
        return test_rmse
