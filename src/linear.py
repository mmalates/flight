from ds_utils import Predictor
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../../data/train.csv')
data_sub, test = train_test_split(data, test_size=0.98)


class Flights(Predictor):

    def processing(self):
        self.data = self.data[list(self.features_) +
                              ['DEPARTURE_DELAY']].dropna()


linear = Flights(data_sub)


linear.dummify(['MONTH', 'DAY_OF_WEEK', 'AIRLINE'])
features = linear.data.drop(['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_ARRIVAL',
                             'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], axis=1).columns

linear.set_features(features)
print list(linear.features_) + ['DEPARTURE_DELAY']
linear.target.describe()
linear.data.info()
print 'processing'
linear.processing()
linear.target = linear.data['DEPARTURE_DELAY']
print 'selecting'
linear.select_features()
len(linear.selected_features_)
len(linear.features_)
