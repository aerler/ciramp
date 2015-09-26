from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
 
class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = GradientBoostingRegressor(n_estimators=100, max_features="sqrt", max_depth=6)
 
    def fit(self, X, y):
        self.clf.fit(X, y.ravel())
 
    def predict(self, X):
        return self.clf.predict(X)
