import numpy as np
import pandas as pd
import time


def compute_all_merge_features():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    player_box = pd.read_pickle('mlb-processed-data/playerBoxScores.pkl')
    # print(next_days.head())
    # print(player_box.head())
    # print(next_days.shape, player_box.shape)
    # print(player_box.columns)
    # player_box has 3 unique keys: playerId, date, gamePk
    return compute_merge_features(next_days, player_box)


# Put this in the Notebook!
def compute_merge_features(next_days, player_box):
    player_box = player_box_features(player_box)
    merged = next_days.merge(player_box, on=['date', 'playerId'], how='left')
    # merged = player_box
    merged = merged.fillna(0)
    # merged = reduce_mem_usage(merged)
    return merged


# Put this in the Notebook!
def player_box_features(player_box):
    # per gamePk, add up features for each playerId, date
    # playerName is constant
    # teamId/teamName is pretty constant (there's only one outlier)
    # jerseyNum has 43 outliers (maybe average them?) and/or take min/max
    # positionCode/Name/Type, battingOrder have ~1000 outliers
    # inningsPitched is a float (0.1)

    player_box = player_box.drop(['gamePk', 'gameDate', 'gameTimeUTC', 'teamId',
       'teamName', 'playerName', 'positionName', 'positionType'], 1)

    # Add number of games played
    player_box['numGames'] = 1

    first = lambda x: x.iloc[0]
    num_columns = len(player_box.columns)
    player_box = player_box.groupby(['date', 'playerId']).agg({'numGames': 'sum',
        'home': 'max', #'gamePk': first, 'gameDate': first, 'gameTimeUTC': first, 'teamId': first,
       #'teamName': first, 'playerName': first,
        'jerseyNum': 'min', 'positionCode': first,
       #'positionName': first, 'positionType': first,
        'battingOrder': 'min', 'gamesPlayedBatting': 'sum',
       'flyOuts': 'sum', 'groundOuts': 'sum', 'runsScored': 'sum', 'doubles': 'sum', 'triples': 'sum', 'homeRuns': 'sum',
       'strikeOuts': 'sum', 'baseOnBalls': 'sum', 'intentionalWalks': 'sum', 'hits': 'sum', 'hitByPitch': 'sum',
       'atBats': 'sum', 'caughtStealing': 'sum', 'stolenBases': 'sum', 'groundIntoDoublePlay': 'sum',
       'groundIntoTriplePlay': 'sum', 'plateAppearances': 'sum', 'totalBases': 'sum', 'rbi': 'sum',
       'leftOnBase': 'sum', 'sacBunts': 'sum', 'sacFlies': 'sum', 'catchersInterference': 'sum',
       'pickoffs': 'sum', 'gamesPlayedPitching': 'sum', 'gamesStartedPitching': 'sum',
       'completeGamesPitching': 'sum', 'shutoutsPitching': 'sum', 'winsPitching': 'sum',
       'lossesPitching': 'sum', 'flyOutsPitching': 'sum', 'airOutsPitching': 'sum',
       'groundOutsPitching': 'sum', 'runsPitching': 'sum', 'doublesPitching': 'sum',
       'triplesPitching': 'sum', 'homeRunsPitching': 'sum', 'strikeOutsPitching': 'sum',
       'baseOnBallsPitching': 'sum', 'intentionalWalksPitching': 'sum', 'hitsPitching': 'sum',
       'hitByPitchPitching': 'sum', 'atBatsPitching': 'sum', 'caughtStealingPitching': 'sum',
       'stolenBasesPitching': 'sum', 'inningsPitched': 'sum', 'saveOpportunities': 'sum',  # inningsPitched is a float (0.1)
       'earnedRuns': 'sum', 'battersFaced': 'sum', 'outsPitching': 'sum', 'pitchesThrown': 'sum', 'balls': 'sum',
       'strikes': 'sum', 'hitBatsmen': 'sum', 'balks': 'sum', 'wildPitches': 'sum', 'pickoffsPitching': 'sum',
       'rbiPitching': 'sum', 'gamesFinishedPitching': 'sum', 'inheritedRunners': 'sum',
       'inheritedRunnersScored': 'sum', 'catchersInterferencePitching': 'sum',
       'sacBuntsPitching': 'sum', 'sacFliesPitching': 'sum', 'saves': 'sum', 'holds': 'sum', 'blownSaves': 'sum',
       'assists': 'sum', 'putOuts': 'sum', 'errors': 'sum', 'chances': 'sum'}).reset_index()
    assert(len(player_box.columns) == num_columns)
    return player_box


def reduce_mem_usage(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def saved_merged(merged):
    merged.to_pickle('mlb-merged-data/merged.pkl')


def main():
    merged = compute_all_merge_features()
    saved_merged(merged)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
