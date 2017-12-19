import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_column', 200)

df = pd.read_csv(
    '/home/mike/projects/flight-delays/data/flights.csv', low_memory=False)

train, test = train_test_split(df, test_size=0.5)

train.to_csv('/home/mike/projects/flight-delays/data/train.csv', index=False)
test.to_csv('/home/mike/projects/flight-delays/data/test.csv', index=False)
