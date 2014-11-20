import numpy as np

class TransitionClassifier:
    def __init__(self,classifier, selected_features):
        self.classifier = classifier
        if len(selected_features) != 1 or selected_features[0] != 'SquaredDifference<RegionCenter>':
            print 'selected_features', selected_features
            raise NotImplementedError, 'other features are not supported yet'
        
    @staticmethod
    def getFeatures(traxel1, traxel2):
        # only squared distances for now
        try:
            feat1 = np.array(traxel1.get_feature_array("RegionCenter"))
            feat2 = np.array(traxel2.get_feature_array("RegionCenter"))
        except:
            raise Exception, 'RegionCenter is not available in traxels'

        return TransitionClassifier.getSquaredDistance(feat1, feat2)

    @staticmethod
    def getSquaredDistance(feat1, feat2):
        return pow(np.linalg.norm( np.array(feat1) - np.array(feat2) ),2.)

    def predict(self, traxel1, traxel2):
        """
        returns probability and variance of transition from Traxel1 to Traxel2
        based on transition classifier (gaussian process classifier)
        """
        x = self.getFeatures(traxel1, traxel2)
        prob, var = self.classifier.predict_probabilities(x, with_variance=True)
        prob = prob.squeeze().tolist()
        var = var.squeeze().tolist()

        return prob, var

    def predictRawWithCoordinates(self, x1, y1, z1, x2, y2, z2):
        return self.predictWithCoordinates(x1, y1, z1, x2, y2, z2, with_raw=True)

    def predictWithCoordinates(self, x1, y1, z1, x2, y2, z2, with_raw=False):
        """
        returns probability and variance of transition from Traxel1 to Traxel2
        based on transition classifier (gaussian process classifier)
        """
        feat1 = np.array([x1, y1, z1])
        feat2 = np.array([x2, y2, z2])
        x = self.getSquaredDistance(feat1, feat2)
        prob, var = self.classifier.predict_probabilities(x, with_variance=True, with_raw=with_raw)
        prob = prob.squeeze().tolist()
        var = var.squeeze().tolist()

        return prob, var

def mk_traxel(x, y, z, id, t, fs):
    traxel = pgmlink.Traxel()
    traxel.ID = id
    traxel.Timestep = t
    traxel.set_feature_store(fs)

    traxel.add_feature_array("com", 3)
    traxel.set_feature_value("com", 0, x)
    traxel.set_feature_value("com", 1, y)
    traxel.set_feature_value("com", 2, z)

    traxel.add_feature_array("divProb",2)
    traxel.set_feature_value("divProb",0, 0.1)
    traxel.set_feature_value("divProb",1, 0.9)

    traxel.add_feature_array("count",1)
    traxel.set_feature_value("count",0, 1)

    return traxel

if __name__ == '__main__':
    import pgmlink
    import h5py
    from lazyflow.classifiers.gaussianProcessClassifier import GaussianProcessClassifier

    ts = pgmlink.TraxelStore()
    fs = pgmlink.FeatureStore()

    t1 = mk_traxel(1,2,3,33,0,fs)
    t2 = mk_traxel(5,6,7,44,1,fs)

    ts.add(fs, t1)
    ts.add(fs, t2)

    fn = '/home/mschiegg/embryonic/tracking/python/tests/test-transition-classifier.h5'
    h5_group = '/TransitionGPClassifier/'

    with h5py.File(fn, 'r') as f:
        g = f[h5_group]
        gpc = GaussianProcessClassifier()
        gpc.deserialize_hdf5(g)

        features = []
        for op in g['Features'].keys():
            for feat in g['Features'][op]:
                features.append('%s<%s>' % (op, feat))

    print 'transition classifier init'
    tc = TransitionClassifier(gpc, features)

    print 'predict'
    res = tc.predict(t1, t2)

    print 'result', res
