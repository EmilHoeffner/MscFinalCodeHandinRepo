# https://medium.com/@prasannarghattikar/using-random-forest-for-feature-importance-118462c40189

from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor


class RandomForestRegressor:

    def __init__(self, max_length = 5):
        self.model = SklearnRandomForestRegressor(n_estimators = 100, max_depth = max_length, random_state = 42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def feature_importances(self):
        return self.model.feature_importances_