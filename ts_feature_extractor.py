import numpy as np

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360-170
en_lon_right = 360-120

class FeatureExtractor(object):
    n_lookahead = -1
    
    def __init__(self):
        pass

    def transform(self, temperatures_xray, n_burn_in, n_lookahead, skf_is):
        self.n_lookahead = n_lookahead
        """Combine two variables: the montly means corresponding to the month of the target and 
        the current mean temperature in the El Nino 3.4 region."""
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        valid_range = range(n_burn_in, temperatures_xray['time'].shape[0] - n_lookahead)
        features = []
        for longitude in xrange(180, 300, 5):
            for lag in xrange(0, 6):
                features.append(self.make_eq_feature(temperatures_xray['tas'], longitude, lag))
        X = np.vstack(features)
        return X.T
    
    def get_enso_mean(self, tas):
        """The array of mean temperatures in the El Nino 3.4 region at all time points."""
        return tas.loc[:, en_lat_bottom:en_lat_top, en_lon_left:en_lon_right].mean(dim=('lat','lon'))

    def get_equatorial_mean(self, tas, x):
        """The array of mean temperatures in the El Nino 3.4 region at all time points."""
        return tas.loc[:, -5:5, x:x+5].mean(dim=('lat','lon'))

    def make_feature(self, enso):
        enso_matrix = enso.values.reshape((-1,12))
        count_matrix = np.ones(enso_matrix.shape)
        enso_monthly_mean = (enso_matrix.cumsum(axis=0) / count_matrix.cumsum(axis=0)).ravel()
        enso_monthly_mean_rolled = np.roll(enso_monthly_mean, self.n_lookahead - 12)
        enso_monthly_mean_valid = enso_monthly_mean_rolled[valid_range]
        enso_valid = enso.values[valid_range]
        return np.array([enso_valid, enso_monthly_mean_valid])

    def make_eq_feature(self, tas, longitude, lag):
        enso = self.get_equatorial_mean(tas, longitude)
        lagged = np.roll(enso, self.n_lookahead - lag)
        return self.make_feature(enso) 
