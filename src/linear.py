from ds_utils import Predictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Flights(Predictor):

    def processing(self, data):
        data = data.dropna(subset=['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE',
                                   'DISTANCE', 'DEPARTURE_DELAY'], axis=0)
        dummies = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE']
        data = self.dummify(data, dummies)
        return data


if __name__ == '__main__':
    print 'loading data'
    data = pd.read_csv('../data/train.csv')
    data_sub, test = train_test_split(data, test_size=0.5)
    test_sub, trash = train_test_split(test, test_size=0.5)
    del test
    del trash

    flights = Flights(data_sub, test_sub, 'DEPARTURE_DELAY')

    print 'processing'
    flights.data = flights.processing(flights.data)

    features = flights.data.drop(['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL',
                                  'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], axis=1).columns
    flights.set_features(features)
    flights.split(test_size=0.5)

    print 'selecting features'
    flights.select_features()

    # print 'tuning hyperparameters'
    # param_grid = {'n_estimators': [10], 'bootstrap': [
    #     True], 'max_features': [0.6], 'verbose': [1]}
    model_name = 'RandomForestRegressor'
    # flights.grid_search(model_name, param_grid)
    # print 'best train RMSE: {}'.format(np.sqrt(abs(flights.train_score_)))

    flights.best_params_ = {
        'bootstrap': True, 'max_features': 0.6, 'n_estimators': 100, 'verbose': 1}
    # score the best model
    print 'testing the performance on unseen data'
    flights.score(model_name, **flights.best_params_)
    print 'test RMSE: {}'.format(flights.test_score_)
    # increase the number of estimators for the final fit
    flights.best_params_['n_estimators'] = 1000

    # fit the best model on the whole dataset
    print 'fitting final model'
    flights.fit(model_name, **flights.best_params_)
    flights.best_params_

    # pickle the model
    print 'pickling model'
    flights.pickle_model('RandomForestRegressor.pkl')

    # predict new data
    print 'making predictions'
    flights.predict()
    flights.data_to_predict['PREDICTED_DELAY'] = flights.predictions
    flights.data_to_predict.to_csv('results.csv', index=False)
    flights.best_params_
