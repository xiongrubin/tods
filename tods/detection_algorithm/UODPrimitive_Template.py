from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
# from numba import njit
from pyod.utils.utility import argmaxn

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer
import uuid

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils

from .UODBasePrimitive import Params_ODBase, Hyperparams_ODBase, UnsupervisedOutlierDetectorBase
from pyod.models.abod import ABOD 
# from typing import Union

Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(Params_ODBase):
    ### Add more Attributes ###

    pass


class Hyperparams(Hyperparams_ODBase):
    ### Add more Hyperparamters ###

    n_neighbors = hyperparams.Hyperparameter[int](
        default=10,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Number of neighbors to use by default for k neighbors queries.",
    )

    method = hyperparams.Enumeration(
        values=['fast', 'default'],
        default='fast',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="'fast': fast ABOD. Only consider n_neighbors of training points    'default': original ABOD with all training points, which could be slow",
    )


### Name of your algorithm ###
class DetectionAlgorithmPrimitive(UnsupervisedOutlierDetectorBase[Inputs, Outputs, Params, Hyperparams]):
    """
    ABOD class for Angle-base Outlier Detection.
    For an observation, the variance of its weighted cosine scores to all
    neighbors could be viewed as the outlying score.
    See :cite:`kriegel2008angle` for details.

    Two versions of ABOD are supported:

    - Fast ABOD: use k nearest neighbors to approximate.
    - Original ABOD: consider all training points with high time complexity at
      O(n^3).

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    n_neighbors : int, optional (default=10)
        Number of neighbors to use by default for k neighbors queries.

    method: str, optional (default='fast')
        Valid values for metric are:

        - 'fast': fast ABOD. Only consider n_neighbors of training points
        - 'default': original ABOD with all training points, which could be
          slow

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

    ### Modify the metadata ###
    # __author__: "DATA Lab at Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({
         "__author__": "DATA Lab at Texas A&M University",
         "name": "Angle-base Outlier Detection Primitive",
         "python_path": "d3m.primitives.tods.detection_algorithm.pyod_abod",
         "source": {
             'name': 'DATA Lab at Texas A&M University', 
             'contact': 'mailto:khlai037@tamu.edu', 
         },
         "hyperparams_to_tune": ['contamination', 'n_neighbors', 'method'],
         "version": "0.0.1",
         "algorithm_types": [
             metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE
         ],
         "primitive_family": metadata_base.PrimitiveFamily.ANOMALY_DETECTION,
	 'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'ABODPrimitive')),
    })

    def __init__(self, *,
                 hyperparams: Hyperparams, #
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        ### Initialize your algorithm ###
        self._clf = DetectionAlgorithm()
                        

    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Set training data for outlier detection.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        super().set_training_data(inputs=inputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """
        return super().fit()

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.

        Returns:
            Container DataFrame
            1 marks Outliers, 0 marks normal.
        """
        return super().produce(inputs=inputs, timeout=timeout, iterations=iterations)

    def produce_score(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.
        Returns:
            Container DataFrame
            Outlier score of input DataFrame.
        """
        return super().produce_score(inputs=inputs, timeout=timeout, iterations=iterations)

    def get_params(self) -> Params:
        """
        Return parameters.
        Args:
            None

        Returns:
            class Params
        """
        return super().get_params()

    def set_params(self, *, params: Params) -> None:
        """
        Set parameters for outlier detection.
        Args:
            params: class Params

        Returns:
            None
        """
        super().set_params(params=params)



### The Implementation of your algorithm ###
class DetectionAlgorithm(BaseDetector):
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

    def __init__(self):
        pass       

    def fit(self, X):
        pass

    def decision_function(self, X):
        pass