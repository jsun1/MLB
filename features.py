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
    merged = reduce_mem_usage(merged)
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


# Computes features that we pre-compute and load into the notebook pre-inference
def compute_pre_features():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    next_days_use = next_days.copy()
    # next_days_use = next_days_use[next_days_use['date'] < '2018-06-01']
    # Convert the date to numeric so it can be passed into group-by
    next_days_use['date'] = pd.to_numeric(next_days_use['date'])
    # Aggregate the cumulative means of the targets, grouped by playerId
    agg = {'date': lambda x: x.values[-1],  # this take the longest time
           'playerId': 'min',
           'target1': 'mean',
           'target2': 'mean',
           'target3': 'mean',
           'target4': 'mean'}
    agg_targets = next_days_use.groupby(['playerId']).expanding().agg(agg).reset_index(drop=True)
    # Shift the targets down by 1, because we can't use the target on its own day (only from previous days)
    shifted = agg_targets.drop(['date'], 1).groupby(['playerId']).shift(1).fillna(0)
    agg_targets[['target1', 'target2', 'target3', 'target4']] = shifted
    # Convert the date back to date-time
    agg_targets['date'] = pd.to_datetime(agg_targets['date'])
    # Rename the columns
    agg_targets = agg_targets.rename(columns={'target1': 'target1_mean', 'target2': 'target2_mean', 'target3': 'target3_mean', 'target4': 'target4_mean'})
    # Merge to sort it back in order
    agg_targets = next_days.merge(agg_targets, on=['date', 'playerId'])
    print(next_days_use.shape)
    print(agg_targets.shape)
    agg_targets = agg_targets[['date', 'playerId', 'target1_mean', 'target2_mean', 'target3_mean', 'target4_mean']]
    agg_targets = reduce_mem_usage(agg_targets)
    agg_targets.to_pickle('mlb-merged-data/pre.pkl')
    return agg_targets


# The very slow iterative early version of compute_pre_features
def compute_pre_features_iter():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    # next_days = next_days.sort_values(by=['date'])  # Don't sort, so that order is same as next_days
    so_far = pd.DataFrame()
    agg_targets = pd.DataFrame()
    a = 0
    current_grouped_mean = None
    # current_grouped_median = None
    current_date = None
    for tuple in next_days.itertuples(index=True):
        index = tuple[0]
        date = tuple[1]
        player_id = tuple[3]
        if date != current_date:
            current_date = date
            if 'playerId' in so_far.columns:
                current_grouped_mean = so_far.groupby(['playerId']).mean()
                # current_grouped_median = so_far.groupby(['playerId']).median()
                # print(current_grouped_mean.head())
                # return
        agg_data = {'date': [date], 'playerId': [player_id],
                    'target1_mean': [0], 'target2_mean': [0], 'target3_mean': [0], 'target4_mean': [0]}#,
                    # 'target1_med': [0], 'target2_med': [0], 'target3_med': [0], 'target4_med': [0]}
        if 'playerId' in so_far.columns and current_grouped_mean is not None:
            so_far_player = so_far[so_far['playerId'] == player_id]
            # grouped = current_grouped[current_grouped['playerId'] == player_id]
            if so_far_player.shape[0] > 1:
                # grouped = so_far_player.groupby(['playerId'])
                so_far_mean = current_grouped_mean.loc[[player_id]]
                agg_data['target1_mean'] = [so_far_mean['target1'].values[0]]
                agg_data['target2_mean'] = [so_far_mean['target2'].values[0]]
                agg_data['target3_mean'] = [so_far_mean['target3'].values[0]]
                agg_data['target4_mean'] = [so_far_mean['target4'].values[0]]
                # so_far_median = current_grouped_median.loc[[player_id]]
                # agg_data['target1_med'] = [so_far_median['target1'].values[0]]
                # agg_data['target2_med'] = [so_far_median['target2'].values[0]]
                # agg_data['target3_med'] = [so_far_median['target3'].values[0]]
                # agg_data['target4_med'] = [so_far_median['target4'].values[0]]
            # else:
            #     print(a)
            #     assert (so_far_player.shape[0] > 1)

        agg_targets = agg_targets.append(pd.DataFrame(data=agg_data), ignore_index=True)
        # Append this row to the "so far" data frame
        so_far = so_far.append(next_days.iloc[index], ignore_index=True)
        if a > 50000:
            break
        a+=1
    # assert(next_days.shape == so_far.shape)
    # assert(next_days.shape[0] == agg_targets.shape[0])
    print('Ensure these are same:', next_days.shape, so_far.shape, agg_targets.shape)
    print(agg_targets.tail())
    # Save to file in reduce memory usage
    # agg_targets = reduce_mem_usage(agg_targets)
    # agg_targets.to_pickle('mlb-merged-data/pre.pkl')


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
    # compute_pre_features()
    merged = compute_all_merge_features()
    # saved_merged(merged)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
