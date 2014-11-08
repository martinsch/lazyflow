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

    def train_binary(self, X, Y):
        assert set(np.unique(Y)) == set([0,1])

        classifier = GPy.models.SparseGPClassification(X, Y, **self._kwargs)
        classifier.tie_params('.*len')

        best = (None, None, None)
        xDim = np.sqrt(X.shape[1])
        lengthscales = [ xDim, xDim / 2., xDim / 4., xDim * 2, xDim * 4, 10, 200 ]


        for lengthscale in lengthscales:
            classifier.kern = GPy.kern.rbf(X.shape[1], lengthscale=lengthscale)
            try:
                classifier.update_likelihood_approximation()
                for i in range(self.parameters['max_iters_ep']):
                    classifier.optimize(max_iters=self.parameters['max_iters_hyperparameters'])

                print 'log likelihood:', classifier.log_likelihood(), 'lengthscale:', lengthscale

                if best[1] is None or classifier.log_likelihood() > best[1]:
                    best = (lengthscale, classifier.log_likelihood(), copy.copy(classifier.getstate()))
            except Exception as e:
                print e
                pass

        print '-----'
        if best[0] is None:
            raise Exception, 'no classifier'

        classifier.setstate(best[2])
        try:
            classifier.update_likelihood_approximation()
            for i in range(self.parameters['max_iters_ep']):
                classifier.optimize(max_iters=self.parameters['max_iters_hyperparameters'])

            print 'log likelihood:', classifier.log_likelihood(), 'initial lengthscale:', best[0]
        except:
            print 'WARNING: could not optimize the best lengthscale'
            pass

        return classifier

    def train_one_vs_all(self, X, Y, nclasses):
        models = []
        assert len(np.unique(Y)) == nclasses

        for c in range(nclasses):
            print
            print 'class', c, 'vs. all'
            Y_class = np.zeros(Y.shape)
            Y_class[Y==(c+1)] = 1  # ilastik classes start at 1
            classifier = self.train_binary(X, Y_class)
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

        if len(X.shape) == 1:
            X = X[..., np.newaxis]

        print np.sum(np.std(X,axis=0)==0), 'of', X.shape[1], 'features are constant'

        if y.ndim == 1:
            y = y[:, numpy.newaxis]

        print X.shape
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

        nclasses = len(known_labels)
        if nclasses == 2:
            y -= 1 # ilastik labels start at 1
            models = [ self.train_binary(X, y) ]
        else:
            models = self.train_one_vs_all(X, y, nclasses)

        return GaussianProcessClassifier( models, known_labels )

    @property
    def description(self):
        return "Gaussian Process Classifier"

assert issubclass( GaussianProcessClassifierFactory, LazyflowVectorwiseClassifierFactoryABC )

class GaussianProcessClassifier(LazyflowVectorwiseClassifierABC):
    """
    Adapt the vigra RandomForest class to the interface lazyflow expects.
    """

    pickle_replace = ("\x00", "!super-awkward replacement hack!")

    def __init__(self, gpcs=None, known_labels=None):
        self._known_labels = known_labels
        self._gpcs = gpcs

    @staticmethod
    def softmax(matrix):
        e = np.exp(np.array(matrix))
        return e / np.sum(e, axis=1, dtype=np.float32)[..., np.newaxis]

    @staticmethod
    def normalize(matrix):
        return np.array(matrix) / np.sum(matrix, axis=1, dtype=np.float32)[..., np.newaxis]

    @staticmethod
    def predict_binary(Xtest, model, with_raw=False):
        if with_raw:
            Xscaled = (Xtest.copy() - model._Xoffset) / model._Xscale
            mu, _var = model.predict(Xscaled)
        pred, var, _, _ = model.predict(numpy.asarray(Xtest, dtype=numpy.float32))

        return pred, var

    @staticmethod
    def predict_one_vs_all(Xtest, models, with_normalization=False):
        Xtest = numpy.asarray(Xtest, dtype=numpy.float32)
        nclasses = len(models)
        predictions = []
        variances = []
        for c in range(nclasses):
            pred, var = GaussianProcessClassifier.predict_binary(Xtest, models[c])
            predictions.append(pred)
            variances.append(var)

        result_pred = np.array(predictions).squeeze().T
        result_var = np.array(variances).squeeze().T

        if with_normalization:
            result_pred = GaussianProcessClassifier.normalize(result_pred)
            assert np.allclose(np.sum(result_pred, axis=1), 1)    # normalization

        assert result_pred.shape == result_var.shape
        assert result_pred.shape == (len(Xtest), len(models))

        return result_pred, result_var


    def predict_probabilities(self, X, with_variance = False):
        logger.debug( 'predicting single-threaded GPy Gaussian Process classifier' )

        if len(self._known_labels) == 2:
            assert len(self._gpcs) == 1
            probs, var = self.predict_binary(X, self._gpcs[0])
            probs = np.hstack((1 - probs, probs))
        else:
            assert len(self._gpcs) == len(self._known_labels)
            probs, var = self.predict_one_vs_all(X, self._gpcs)
            print 'WARNING: var is per binary GPC'

        probs = probs
        var = var

        if with_variance:
            return probs, var
        return probs
    
    @property
    def known_classes(self):
        return self._known_labels
    
    def serialize_hdf5(self, h5py_group):
        for i, gpc in enumerate(self._gpcs):
            pickled = gpc.pickles()
            pickled = pickled.replace(*self.pickle_replace)
            h5py_group.create_dataset("GPCpickle_%03d" % i, data=pickled)
            try:
                h5py_group.create_dataset("known_labels", data=self._known_labels)
            except:
                pass
        
    def deserialize_hdf5(self, h5py_group):
        # assert "GPCpickle" in h5py_group
        gpcs = []
        for k in sorted(h5py_group.keys()):
            if "GPCpickle" not in k:
                continue
            s = h5py_group[k].value
            s = s.replace(*(self.pickle_replace[::-1]))
            assert len(gpcs) == int(k.split('_')[-1])
            gpcs.append(pickle.loads(s))

        self._gpcs = gpcs
        self._known_labels = np.array(h5py_group["known_labels"]).tolist()

        return gpcs

assert issubclass( GaussianProcessClassifier, LazyflowVectorwiseClassifierABC )
