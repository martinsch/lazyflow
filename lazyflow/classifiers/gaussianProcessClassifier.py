import os
import tempfile
import cPickle as pickle
from GPy import kern
import numpy
import h5py
import copy

import GPy
from .lazyflowClassifier import LazyflowVectorwiseClassifierABC, LazyflowVectorwiseClassifierFactoryABC

import logging
import numpy as np

logger = logging.getLogger(__name__)

class GaussianProcessClassifierFactory(LazyflowVectorwiseClassifierFactoryABC):
    parameters = { "max_iters_hyperparameters": 50,
                   "max_iters_ep": 5,
                   "max_iters_initializations": 1,
                   "num_inducing": 20,
                   "kernel": 'rbf',
                   "normalize_X": True,
                   "normalize_Y": False,
                   # "lengthscales": [200, 0.5,1,50]
                   }

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def train_one_vs_all(self, X, Y, nclasses):
        models = []
        assert len(np.unique(Y)) == nclasses

        for c in range(nclasses):
            print
            print 'class', c, 'vs. all'
            Y_class = np.zeros(Y.shape)
            Y_class[Y==c] = 1
            classifier = GPy.models.SparseGPClassification(X, Y_class, **self._kwargs)
            classifier.tie_params('.*len')

            best = (None, None, None)
            xDim = np.sqrt(X.shape[1])
            lengthscales = [ xDim, xDim / 2., xDim / 4., xDim * 2, xDim * 4 ]

            for lengthscale in lengthscales:
                classifier.kern = GPy.kern.rbf(X.shape[1], lengthscale=lengthscale)
                classifier.update_likelihood_approximation()

                for i in range(self.parameters['max_iters_ep']):
                    classifier.optimize(max_iters=self.parameters['max_iters_hyperparameters'])

                print 'log likelihood:', classifier.log_likelihood(), 'lengthscale:', lengthscale

                if best[1] is None or classifier.log_likelihood() > best[1]:
                    best = (lengthscale, classifier.log_likelihood(), copy.copy(classifier.getstate()))

            print '-----'
            classifier.setstate(best[2])
            classifier.update_likelihood_approximation()
            for i in range(self.parameters['max_iters_ep']):
                classifier.optimize(max_iters=self.parameters['max_iters_hyperparameters'])

            print 'log likelihood:', classifier.log_likelihood(), 'initial lengthscale:', best[0]

            models.append(classifier)

        return models


    def create_and_train(self, X, y):
        logger.debug( 'training single-threaded GaussianProcessClassifier' )
        
        # Save for future reference
        known_labels = numpy.unique(y)

        X = numpy.asarray(X, numpy.float32)
        y = numpy.asarray(y, numpy.uint32)

        if np.isnan(X).any():
            print 'replacing nans by zeros'
            X[np.isnan(X)] = 0

        print np.sum(np.std(X,axis=0)==0), 'of', X.shape[1], 'features are constant'

        if y.ndim == 1:
            y = y[:, numpy.newaxis]

        assert X.ndim == 2
        assert len(X) == len(y)

        if self.parameters['kernel'] == 'rbf':
            self._kwargs["kernel"] = kern.rbf(X.shape[1])
        elif self.parameters['kernel'] == 'rbf_ard':
            self._kwargs["kernel"] = kern.rbf(X.shape[1], ARD=True)
        else:
            raise NotImplementedError, 'this kernel is not supported yet'

        self._kwargs["normalize_X"] = bool(self.parameters['normalize_X'])
        self._kwargs["normalize_Y"] = bool(self.parameters['normalize_Y'])
        self._kwargs["num_inducing"] = self.parameters['num_inducing']

        models = self.train_one_vs_all(X, y, len(known_labels))

        return GaussianProcessClassifier( models, known_labels )

    @property
    def description(self):
        return "Gaussian Process Classifier"

assert issubclass( GaussianProcessClassifierFactory, LazyflowVectorwiseClassifierFactoryABC )

class GaussianProcessClassifier(LazyflowVectorwiseClassifierABC):
    """
    Adapt the vigra RandomForest class to the interface lazyflow expects.
    """
    def __init__(self, gpcs, known_labels):
        self._known_labels = known_labels
        self._gpcs = gpcs


    @staticmethod
    def softmax(matrix):
        e = np.exp(np.array(matrix))
        return e / np.sum(e, axis=0)

    @staticmethod
    def predict_one_vs_all(Xtest, models, with_raw=False):
        Xtest = numpy.asarray(Xtest, dtype=numpy.float32)
        nclasses = len(models)
        predictions = []
        variances = []
        for c in range(nclasses):
            if with_raw:
                Xscaled = (Xtest.copy() - models[c]._Xoffset) / models[c]._Xscale
                mu, _var = models[c].predict(Xscaled)
            pred, var, _, _ = models[c].predict(numpy.asarray(Xtest, dtype=numpy.float32))
            predictions.append(pred)
            variances.append(var)
        pred_softmax = GaussianProcessClassifier.softmax(np.array(predictions)).squeeze().T
        var_softmax = GaussianProcessClassifier.softmax(np.array(variances)).squeeze().T

        assert pred_softmax.shape == var_softmax.shape
        assert pred_softmax.shape == (len(Xtest), len(models))
        assert np.allclose(np.sum(pred_softmax, axis=1), 1) # normalization

        return pred_softmax, var_softmax


    def predict_probabilities(self, X, with_variance = False):
        logger.debug( 'predicting single-threaded GPy Gaussian Process classifier' )

        probs, var = self.predict_one_vs_all(X, self._gpcs)

        if with_variance:
            print 'WARNING: var is per binary GPC'
            return probs, var
        return probs
    
    @property
    def known_classes(self):
        return self._known_labels
    
    def serialize_hdf5(self,h5py_group):
        for i, gpc in enumerate(self._gpcs):
            pickled = gpc.pickles()
            pickled = pickled.replace("\x00","!super-awkward replacement hack!")
            h5py_group.create_dataset("GPCpickle_%03d" % i, data=pickled)
        
    def deserialize_hdf5(self,h5py_group):
        # assert "GPCpickle" in h5py_group
        gpcs = []
        for k in sort(h5py_group.keys()):
            if "GPCpickle" not in k:
                continue
            s = h5py_group[k]
            s = s.replace("!super-awkward replacement hack!", "\x00")
            assert len(gpcs) == int(k.split('_')[-1])
            gpcs.append(pickle.loads(s))

        return gpcs

assert issubclass( GaussianProcessClassifier, LazyflowVectorwiseClassifierABC )
