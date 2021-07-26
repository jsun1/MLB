import json
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time

# Read training data
# training = pd.read_csv(data_dir / 'train.csv', usecols=['date'] + dfs)


def read_train():
    training = pd.read_csv('mlb-player-digital-engagement-forecasting/train_updated.csv',
                           usecols=['date', 'transactions', 'standings', 'awards', 'events', 'playerTwitterFollowers',
                                    'teamTwitterFollowers', 'nextDayPlayerEngagement', 'playerBoxScores', 'rosters',
                                    'games', 'teamBoxScores'])
    # date is a unique key
    #TODO: is setting the date to a datetime needed?
    training['date'] = pd.to_datetime(training['date'], format='%Y%m%d')
    print(training.shape)
    print(training.columns)
    # print(training.head())
    print(training.tail())
    print(training['date'].nunique())
    # training['date'].plot.hist(bins=20)
    # plt.show()
    # print(training['nextDayPlayerEngagement'][0])
    # print(training['playerBoxScores'][training['playerBoxScores'].notnull()])
    # verify_dates(training, 'transactions')
    columns = training.columns
    for column in columns:
        if column != 'date':
            un_nested_form = un_nest(training, column)
            un_nested_form.to_pickle('mlb-processed-data/' + column + '.pkl')


# Put this in the Notebook!
def un_nest(full_data, field):
    # create the data frame (un nested) for each field
    next_day_training = full_data[['date', field]]
    next_days = pd.DataFrame()
    #TODO: use apply to read json faster?
    for tuple in next_day_training.itertuples(index=False):
        date = tuple[0]
        next_day_string = tuple[1]
        if isinstance(next_day_string, str):
            next_day = pd.read_json(next_day_string, orient='records')
            if 'date' in next_day.columns:
                next_day = next_day.drop('date', 1)
            next_day.insert(0, 'date', date)  # add the date to each data frame
            next_days = next_days.append(next_day)
        else:
            assert(math.isnan(next_day_string))
    return next_days


def read_pickles():
    # Reads the saved pickle files to verify some info
    field = 'transactions'
    df = pd.read_pickle('mlb-processed-data/' + field + '.pkl')
    print(field)
    print(df.columns)
    # print(df.head())
    # print(df.tail())
    assert('date' not in df.columns)


def verify_dates(training, field):
    # ensures the 'date' is the same in the data frame and sub data frames
    next_day_training = training[['date', field]]
    next_days = pd.DataFrame()
    for tuple in next_day_training.itertuples(index=False):
        date = tuple[0]
        next_day_string = tuple[1]
        # next_day = pd.read_json(next_day_string, orient='records')
        # next_days = next_days.append(next_day)
        # print(next_day_string, type(next_day_string))
        if isinstance(next_day_string, str):
            next_day = pd.read_json(next_day_string, orient='records')
            next_days = next_days.append(next_day)
            next_day['date'] = pd.to_datetime(next_day['date'], format='%Y-%m-%d')
            for i, row in next_day.iterrows():
                assert(date == row['date'])
        else:
            assert(math.isnan(next_day_string))
        # print(next_day.head())
    print(field, next_days.shape)
    # next_days.to_pickle('mlb-processed-data/' + field + '.pkl')


def main():
    read_train()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
