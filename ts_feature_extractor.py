        # the burn-in from the beginning and the prediction look-ahead from
        # the end.
        self.valid_range = range(n_burn_in, temperatures_xray['time'].shape[0] - n_lookahead)
        features = []
        for latitude in xrange(-60, 60, 5):
            for longitude in xrange(180, 300, 5):
                for lag in xrange(0, 6):
                    features.append(self.make_ll_feature(temperatures_xray['tas'], latitude, longitude, lag))
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
        enso_monthly_mean_valid = enso_monthly_mean_rolled[self.valid_range]
        enso_valid = enso.values[self.valid_range]
        return np.array([enso_valid, enso_monthly_mean_valid])

    def make_eq_feature(self, tas, longitude, lag):
        enso = self.get_equatorial_mean(tas, longitude)
        lagged = np.roll(enso, self.n_lookahead - lag)
        return self.make_feature(enso) 

    def make_ll_feature(self, tas, latitude, longitude, lag):
        enso = tas.loc[:, latitude:latitude+10, longitude:longitude+10].mean(dim=('lat','lon'))
        lagged = np.roll(enso, self.n_lookahead - lag)
        return self.make_feature(enso) 
