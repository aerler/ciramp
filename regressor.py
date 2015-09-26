from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA

class Regressor(BaseEstimator):
    def __init__(self):
        #self.clf = GradientBoostingRegressor(n_estimators=100, max_features="sqrt", max_depth=6)
        self.clf = make_pipeline(
                StandardScaler(),
                SparsePCA(),
                GradientBoostingRegressor(n_estimators=200, max_features="sqrt", max_depth=5)
        )

        
    def fit(self, X, y):
        self.clf.fit(X, y.ravel())
 
    def predict(self, X):
        return self.clf.predict(X)
