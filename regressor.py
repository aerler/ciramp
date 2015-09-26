from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
 
class Regressor(BaseEstimator):
    def __init__(self):
        #self.clf = GradientBoostingRegressor(n_estimators=100, max_features="sqrt", max_depth=6)
        self.clf = make_pipeline(
                StandardScaler(),
                KernelPCA(n_components=600, kernel='poly', degree=3),
                GradientBoostingRegressor(n_estimators=200, max_features="sqrt", max_depth=5)
        )

        
    def fit(self, X, y):
        self.clf.fit(X, y.ravel())
 
    def predict(self, X):
        return self.clf.predict(X)
