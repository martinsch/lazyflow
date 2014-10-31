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
    
    def create_and_train(self, X, y):
        logger.debug( 'training single-threaded GaussianProcessClassifier' )
        
        # Save for future reference
        known_labels = numpy.unique(y)
        
        X = numpy.asarray(X, numpy.float32)
        y = numpy.asarray(y, numpy.uint32)

        print np.sum(np.std(X,axis=0)==0), 'of', X.shape[1], 'features are constant'

        if y.ndim == 1:
            y = y[:, numpy.newaxis]

        y-=1 # FIXME: for the binary case, we need classes 0 and 1 rather than 1 and 2. Fix for the multi-label case
        assert X.ndim == 2
        assert len(X) == len(y)
        assert all(i in (0,1) for i in numpy.unique(y))

        if self.parameters['kernel'] == 'rbf':
            self._kwargs["kernel"] = kern.rbf(X.shape[1])
        elif self.parameters['kernel'] == 'rbf_ard':
            self._kwargs["kernel"] = kern.rbf(X.shape[1], ARD=True)
        else:
            raise NotImplementedError, 'this kernel is not supported yet'

        self._kwargs["normalize_X"] = bool(self.parameters['normalize_X'])
        self._kwargs["normalize_Y"] = bool(self.parameters['normalize_Y'])
        self._kwargs["num_inducing"] = self.parameters['num_inducing']

        classifier = GPy.models.SparseGPClassification(X,
                                                       y,
                                                        **self._kwargs)

        # constrain all lengthscale parameters to be positive
        classifier.tie_params('.*len')
        # classifier.update_likelihood_approximation()
        # classifier.ensure_default_constraints()

        best = (None, None, None)

        xDim = np.sqrt(X.shape[1])
        lengthscales = [ xDim, xDim / 2., xDim / 4., xDim * 2, xDim * 4 ]

        for lengthscale in lengthscales:
            # for j in range(self.parameters['max_iters_initializations']):
            #     print 'randomize initialization'
            #     classifier.randomize()
            classifier.kern = kern.rbf(X.shape[1], lengthscale=lengthscale)
            classifier.update_likelihood_approximation()

            for i in range(self.parameters['max_iters_ep']):
                print 'ep iteration'
                classifier.optimize(max_iters=self.parameters['max_iters_hyperparameters'])

            print 'log likelihood:', classifier.log_likelihood(), 'lengthscale:', lengthscale
            if best[1] is None or classifier.log_likelihood() > best[1]:
                print 'that is better, setting best to', lengthscale
                best = (lengthscale, classifier.log_likelihood(), copy.copy(classifier.getstate()))
        print '-----'

        # classifier.kern = kern.rbf(X.shape[1], lengthscale=best[0])
        classifier.setstate(best[2])

        classifier.update_likelihood_approximation()
        # classifier.optimize(max_iters=1)
        for i in range(self.parameters['max_iters_ep']):
            classifier.optimize(max_iters=self.parameters['max_iters_hyperparameters'])

        print 'log likelihood:', classifier.log_likelihood(), 'lengthscale:', best[0]

        return GaussianProcessClassifier( classifier, known_labels )

    @property
    def description(self):
        return "Gaussian Process Classifier"

assert issubclass( GaussianProcessClassifierFactory, LazyflowVectorwiseClassifierFactoryABC )

class GaussianProcessClassifier(LazyflowVectorwiseClassifierABC):
    """
    Adapt the vigra RandomForest class to the interface lazyflow expects.
    """
    def __init__(self, gpc, known_labels):
        self._known_labels = known_labels
        self._gpc = gpc
    
    def predict_probabilities(self, X, with_variance = False):
        logger.debug( 'predicting single-threaded vigra RF' )
        
        X = numpy.asarray(X, dtype=numpy.float32)
        Xnew = (X.copy() - self._gpc._Xoffset) / self._gpc._Xscale
        mu, _var = self._gpc._raw_predict(Xnew)
        
        probs,var,_,_ = self._gpc.predict(numpy.asarray(X, dtype=numpy.float32) )
        
        #here, mu == inverse_sigmoid(probs) == GPy.util.univariate_Gaussian.inv_std_norm_cdf(probs)*numpy.sqrt(1+_var)
        
        #we get the probability p for label 1 here,
        #so we complete the table by adding the probability for label 0, which is 1-p
        if with_variance:
            return numpy.concatenate((1-probs,probs),axis = 1),_var
        return  numpy.concatenate((1-probs,probs),axis = 1)
    
    @property
    def known_classes(self):
        return self._known_labels
    
    def serialize_hdf5(self,h5py_group):
        pickled = self._gpc.pickles()
        pickled = pickled.replace("\x00","!super-awkward replacement hack!")
        h5py_group.create_dataset("GPCpickle",data=pickled)
        
    def deserialize_hdf5(self,h5py_group):
        assert "GPCpickle" in h5py_group
        s = h5py_group["GPCpickle"]
        s = s.replace("!super-awkward replacement hack!","\x00")
        classifier = pickle.loads(s)
        return classifier

assert issubclass( GaussianProcessClassifier, LazyflowVectorwiseClassifierABC )
