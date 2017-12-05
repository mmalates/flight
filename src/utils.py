import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.ensemble as en


class Predictor(object):
    """Process data, train models, and predict flight delays"""

    def __init__(self, data=None):
        """Reads in data and initializes some attributes for later

        Args:
            data: preloaded dataframe, default is None
        """
        self.data = data
        self.target = None
        self.model_dict = {'Linear': lm.LinearRegression(),
                           'Lasso': lm.Lasso(),
                           'Ridge': lm.Ridge,
                           'RandomRorest': en.RandomForestRegressor(),
                           'AdaBoost': en.AdaBoostRegressor(),
                           'GradientBoost': en.GradientBoostingRegressor(),
                           'Bagging': en.BaggingRegressor()}
        self.features_ = []
        self.selected_features_ = []
        self.model = None

    def load_data(self, filepath, sep=","):
        """loads csv or json data from file

        Args:
            filepath (str): full or relative path to file being loaded
            sep (str): delimiter for csv file being loaded, default is ","
        """
        if filename.split('.')[1] == 'csv':
            self.data = pd.read_csv(filepath, sep=sep)
        if filename.split('.')[1] == 'json':
            self.data = pd.read_json(filepath)
        else:
            print 'Please select a csv or json file'

    def processing(self):
        self.data.dropna(inplace=True)
        pass

    def fit(self, model_name, features=self.selected_features_, target, **model_params):
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
        self.target = self.data[target].fillna(self.data[target].median())
        self.data = self.data[features].fillna(self.data[features].median())
        model = self.model_dict[model_name]
        model.set_params(**model_params)
        self.model = model.fit(self.data, self.target)

    def set_features(self, features):
        self.features_ = features

    def select_features(self):
        """trains a Lasso regression and drops features with 0 coefficients"""
        model = lm.LassoCV(normalize=True)
        model.fit(self.data[self.features], self.target)
        with open('lasso_coefficients.txt', 'w') as f:
            for coef, feature in sorted(zip(trained_model.coef_, self.train_features_$
                f.write('{} : {}\n'.format(feature, coef))
                if coef not in [-0.0, 0.0]:
                    self.selected_features_.append(feature)

    def predict(self, data_to_predict):
        return self.model.predict(data_to_predict)

    def dummify(self, dummy_columns):
        """Dummifies categorical features

        Args:
            dummy_columns (list): columns to create dummy columns from
        """
        for column in dummy_columns:
            dummies=pd.get_dummies(self.data[column], prefix=column)
            self.data=pd.concat((self.data, dummies), axis=1)
            self.data.drop(column, axis=1, inplace=True)


if __name__ == '__main__':
    """load data
    process data
    dummify
    set features
    select features
    fit model
    predict
    """
    pass
