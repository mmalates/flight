import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.ensemble as en


class Predictor(object):
    """Process data, train models, and predict flight delays"""

    def __init__(self, data_file,):
        """Reads in data and initializes some attributes for later"""
        self.data = pd.read_csv(data_file)
        self.model_dict = {'Linear': lm.LinearRegression(),
                           'Lasso': lm.Lasso(),
                           'Ridge': lm.Ridge,
                           'RandomRorest': en.RandomForestRegressor(),
                           'AdaBoost': en.AdaBoostRegressor(),
                           'GradientBoost': en.GradientBoostingRegressor(),
                           'Bagging': en.BaggingRegressor()}
        self.features = []
        self.model = None

    def processing(self):
        pass

    def train(self, model_name, target, **model_params):
        """Train model on training data

        Args:
            model_name (str): options are
                                {'Linear': lm.LinearRegression(),
                                'Lasso': lm.Lasso(),
                                'Ridge': lm.Ridge,
                                'RandomRorest': en.RandomForestRegressor(),
                                'AdaBoost': en.AdaBoostRegressor(),
                                'GradientBoost': en.GradientBoostingRegressor(),
                                'Bagging': en.BaggingRegressor()}
            **model_params: Parameters to be passed to model
        Returns:
            trained model
        """
        if model_name == 'Linear':
            for col in self.data.drop.columns:
                if pd.api.types.is_numeric_dtype(self.data.col) & col != target:
                    self.features.append(col)
        model = self.model_dict[model_name]
        model.set_params(**model_params)
        self.model = model.fit(self.data[self.features], self.data[target])

    def predict(self, data_to_predict):
        return self.model.predict(data_to_predict)

    def dummify(self, dummy_columns):
        """Dummifies categorical features

        Args:
            dummy_columns (list): columns to create dummy columns from
        """
        for column in dummy_columns:
            dummies = pd.get_dummies(self.data[column], prefix=column)
            self.data = pd.concat((self.data, dummies), axis=1)
            self.data.drop(column, axis=1, inplace=True)


if __name__ == '__main__':
    pass
