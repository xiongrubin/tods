# import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
# from .CollectiveBase import CollectiveBaseDetector
from numpy import percentile
import autokeras as ak
from tods.detection_algorithm.core.ak.blocks import RNNBlock
from tods.detection_algorithm.core.ak.heads import ReconstructionHead


# from  .compression_net import CompressionNet
# from .estimation_net import EstimationNet
# from .gmm import GMM
# from pyod.utils.stat_models import pairwise_distances_no_broadcast

from os import makedirs
from os.path import exists, join
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import d3m

from pyod.models.base import BaseDetector

### The Implementation of your algorithm ###
class AKRNN(BaseDetector):
    """
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.
    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, batch_size:int = 32, 
                    epochs:int = 10, 
                    validation_split:float = 0.2,
                    contamination = 0.1
                    ):

        # parameter initialization
        super(AKRNN, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.contamination = contamination
        

    def fit(self,X,y=None):
        if isinstance(X, d3m.container.pandas.DataFrame):
            X = X.to_numpy()
        # X = X.to_numpy() #assume input is d3m container df

        X = np.expand_dims(   #this is for convblock and rnn
            X, axis=1
        ) #ASK if this is needed

        # ak model initialization
        shape = X.shape[-1]
        inputs = ak.Input(shape=[shape,])
        mlp_output = RNNBlock()([inputs])
        output = ReconstructionHead()(mlp_output)

        self.auto_model = ak.AutoModel(inputs=inputs,
                          outputs=output, 
                          project_name="auto_model_ak_rnn",
                          objective='val_mean_squared_error',
                          max_trials=1
                          )
        self.auto_model.fit(x=[X],
               y=X,
               batch_size=128,
               epochs=2)

        pred = self.auto_model.predict(x=[X])

        self.decision_scores_ = np.sqrt(np.sum(X - pred, axis=1)).ravel()

        self.threshold_ = percentile(self.decision_scores_,
                                     100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(
            'int').ravel()

        # self._process_decision_scores()
        # return self


    def decision_function(self, X):
        pred = self.auto_model.predict(x=[X])
        mse = np.sqrt(np.sum(X - pred, axis=1)).ravel()
        print('mse--------',mse)
        return mse
