from ds_utils import Predictor
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('../../data/train.csv')
data, test = train_test_split(data, test_size=0.9)


linear = Flights(data)

linear.data.columns
linear.dummify(['MONTH', 'DAY_OF_WEEK', 'AIRLINE',
                'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
linear.set_features(linear.data.drop(['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON',
                                      'TAXI_IN', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], axis=1).columns)

linear.target = linear.data['DEPARTURE_DELAY']
linear.processing()
linear.select_features()


class Flights(Predictor):

    def processing(self):
        self.data.dropna(
            inplace=True, subset=self.features_.append('DEPARTURE_DELAY'))
        pass
