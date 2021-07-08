import numpy as np
import pandas as pd
import time


def compute_all_merge_features():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    player_box = pd.read_pickle('mlb-processed-data/playerBoxScores.pkl')
    pre_agg = pd.read_pickle('mlb-merged-data/pre_train.pkl')
    year = pd.read_pickle('mlb-merged-data/years.pkl')
    month = pd.read_pickle('mlb-merged-data/months.pkl')
    day = pd.read_pickle('mlb-merged-data/days_of_week.pkl')
    # print(next_days.head())
    # print(player_box.head())
    # print(next_days.shape, player_box.shape)
    # print(player_box.columns)
    # player_box has 3 unique keys: playerId, date, gamePk
    return compute_merge_features(next_days, player_box, pre_agg, year, month, day, True)


# Put this in the Notebook!
def compute_merge_features(next_days, player_box, pre_agg, year, month, day, training):
    if training:
        # Limit training only to the season
        # next_days = next_days[(next_days['date'] > '2018-03-29') & (next_days['date'] <= '2018-10-01') |
        next_days = next_days[(next_days['date'] > '2019-03-20') & (next_days['date'] < '2019-09-29') |
                              (next_days['date'] > '2020-07-23') & (next_days['date'] < '2020-09-27') |
                              (next_days['date'] > '2021-04-01') & (next_days['date'] <= '2021-04-30')]
    player_box = player_box_features(player_box)
    merged = next_days.merge(player_box, on=['date', 'playerId'], how='left')
    merged = merged.fillna(0)  # this will fill player_box as 0 for when there was no game played
    merged = merge_date_features(merged, year, month, day, training)
    merged = merge_pre_agg(merged, pre_agg, training)
    if not training:
        merged = merged.fillna(0)
    # Dataset normalization
    not_feature = ['date', 'playerId', 'jerseyNum', 'target1', 'target2', 'target3', 'target4']
    if training:
        not_feature.append('engagementMetricsDate')
    norm = merged.drop(not_feature, 1)
    if training:
        norm_min = norm.min()
        norm_std = norm.std()
        norm_min.to_pickle('mlb-merged-data/norm_min.pkl', protocol=4)
        norm_std.to_pickle('mlb-merged-data/norm_std.pkl', protocol=4)
    else:
        norm_min = pd.read_pickle('../input/mlbmergeddata/norm_min.pkl')
        norm_std = pd.read_pickle('../input/mlbmergeddata/norm_std.pkl')
    norm = (norm - norm_min) / norm_std
    features = [f for f in list(merged.columns) if f not in not_feature]
    merged[features] = norm
    # merged = reduce_mem_usage(merged)
    return merged


# Put this in the Notebook!
def merge_date_features(merged, year, month, day, training):
    if not training:
        merged['date'] = pd.to_datetime(merged['date'])
    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month
    merged['dayOfWeek'] = merged['date'].dt.dayofweek
    # Year
    merged = merged.merge(year, on='year', how='left')
    for i in ['1', '2', '3', '4']:
        merged['year_tg' + i] = np.where(merged['numGames'] == 0, merged['year_tn' + i], merged['year_ty' + i])
    # Month
    merged = merged.merge(month, on='month', how='left')
    for i in ['1', '2', '3', '4']:
        merged['month_tg' + i] = np.where(merged['numGames'] == 0, merged['month_tn' + i], merged['month_ty' + i])
    # Day of week
    merged = merged.merge(day, on='dayOfWeek', how='left')
    for i in ['1', '2', '3', '4']:
        merged['day_tg' + i] = np.where(merged['numGames'] == 0, merged['day_tn' + i], merged['day_ty' + i])
    to_drop = ['year', 'month', 'dayOfWeek']
    for name in ['year', 'month', 'day']:
        for i in ['1', '2', '3', '4']:
            to_drop.append(name + '_ty' + i)
            to_drop.append(name + '_tn' + i)
    merged = merged.drop(to_drop, 1)
    return merged


# Add linear date features (they didn't help with the training/validation loss)
def merge_date_features_linear(merged, next_days):
    dates = next_days[['date']].copy().reset_index(drop=True)
    dates['year'] = dates['date'].dt.year.apply(lambda x: 0 if x <= 2019 else 1)
    dates['month'] = dates['date'].dt.month
    dates['day'] = dates['date'].dt.day
    dates['dayOfWeek'] = dates['date'].dt.dayofweek
    dates['week'] = dates['date'].dt.isocalendar().week
    dates = dates.drop(['date'], 1)
    # merged = merged.reset_index(drop=True).join(dates)
    merged[['year', 'month', 'day', 'dayOfWeek', 'week']] = dates
    print("MMM")
    print(merged.tail())
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
    # Add a column for total number of players that have games on a given day (not helpful)
    # total = player_box[['date', 'numGames']].groupby(['date']).sum().rename(columns={'numGames': 'totalPlayerGames'})
    # player_box = player_box.merge(total, on='date', how='left')
    return player_box


# Put this in the Notebook!
def merge_pre_agg(merged, pre_agg, training):
    # From the pre_train/test.pkl aggregations, compute just the final values
    assert(merged['numGames'].isna().sum() == 0)
    m_shape = merged.shape
    if training:
        # In training, use all values
        merged = merged.merge(pre_agg, on=['date', 'playerId'])
    else:
        # In testing, use only final values
        # Account for unknown players
        # TODO: this still gives nan on submission
        is_known = merged['playerId'].isin(pre_agg['playerId'])
        new_players = pd.DataFrame(is_known, columns=['playerId']).rename(columns={'playerId': 'is_known'})
        new_players['playerId'] = merged['playerId']
        new_players = new_players[(new_players['is_known'] == False)].drop(['is_known'], 1).reset_index(drop=True)
        lookup = pre_agg
        if new_players.shape[0] > 0:
            # If there are new players, add the unknown player template to the lookup
            unknown = pre_agg[pre_agg['playerId'] == -1].reset_index(drop=True).copy()
            unknown = unknown.loc[unknown.index.repeat(new_players.shape[0])].reset_index(drop=True)
            unknown['playerId'] = new_players['playerId']
            lookup = lookup.append(unknown, ignore_index=True)
        # Assume pre_agg is a lookup data frame which can be merged via playerId
        merged = merged.merge(lookup, on=['playerId'], how='left')
    # Use the yes/no game target depending on numGames
    merged['target1_game'] = np.where(merged['numGames'] == 0, merged['target1_n_game'], merged['target1_y_game'])
    merged['target2_game'] = np.where(merged['numGames'] == 0, merged['target2_n_game'], merged['target2_y_game'])
    merged['target3_game'] = np.where(merged['numGames'] == 0, merged['target3_n_game'], merged['target3_y_game'])
    merged['target4_game'] = np.where(merged['numGames'] == 0, merged['target4_n_game'], merged['target4_y_game'])
    merged['target1_game_mean'] = np.where(merged['numGames'] == 0, merged['target1_n_game_mean'], merged['target1_y_game_mean'])
    merged['target2_game_mean'] = np.where(merged['numGames'] == 0, merged['target2_n_game_mean'], merged['target2_y_game_mean'])
    merged['target3_game_mean'] = np.where(merged['numGames'] == 0, merged['target3_n_game_mean'], merged['target3_y_game_mean'])
    merged['target4_game_mean'] = np.where(merged['numGames'] == 0, merged['target4_n_game_mean'], merged['target4_y_game_mean'])
    merged = merged.drop(['target1_y_game', 'target2_y_game', 'target3_y_game', 'target4_y_game',
                          'target1_n_game', 'target2_n_game', 'target3_n_game', 'target4_n_game',
                          'target1_y_game_mean', 'target2_y_game_mean', 'target3_y_game_mean', 'target4_y_game_mean',
                          'target1_n_game_mean', 'target2_n_game_mean', 'target3_n_game_mean', 'target4_n_game_mean'
                          ], 1)
    assert(merged.shape[0] == m_shape[0])
    return merged


# Computes features that we pre-compute and load into the notebook pre-inference
def compute_pre_features(training=True):
    next_days_orig = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    # Limit computation to only the season and the preceding month
    # (next_days_orig['date'] > '2018-02-28') & (next_days_orig['date'] <= '2018-10-01') |
    next_days_orig = next_days_orig[(next_days_orig['date'] > '2019-02-20') & (next_days_orig['date'] < '2019-09-29') |
                                    (next_days_orig['date'] > '2020-06-23') & (next_days_orig['date'] < '2020-09-27') |
                                    (next_days_orig['date'] > '2021-03-01') & (next_days_orig['date'] <= '2021-04-30')]
    player_box = player_box_features(pd.read_pickle('mlb-processed-data/playerBoxScores.pkl'))
    merged = next_days_orig.merge(player_box, on=['date', 'playerId'], how='left')
    merged = merged.fillna(0)  # this will fill player_box as 0 for when there was no game played
    next_days = merged[['date', 'playerId', 'numGames', 'target1', 'target2', 'target3', 'target4']].copy()
    # next_days_use = next_days_use[next_days_use['date'] < '2018-06-01']
    # Convert the date to numeric so it can be passed into group-by
    next_days['date'] = pd.to_numeric(next_days['date'])  # next_days = next_days.assign(date=pd.to_numeric(next_days['date']))
    next_days[['target1_med', 'target2_med', 'target3_med', 'target4_med']] = next_days[['target1', 'target2', 'target3', 'target4']]
    next_days[['target1_mean', 'target2_mean', 'target3_mean', 'target4_mean']] = next_days[['target1', 'target2', 'target3', 'target4']]
    next_days['target1_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target1'])
    next_days['target2_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target2'])
    next_days['target3_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target3'])
    next_days['target4_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target4'])
    next_days['target1_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target1'])
    next_days['target2_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target2'])
    next_days['target3_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target3'])
    next_days['target4_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target4'])
    next_days['target1_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target1'])
    next_days['target2_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target2'])
    next_days['target3_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target3'])
    next_days['target4_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target4'])
    next_days['target1_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target1'])
    next_days['target2_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target2'])
    next_days['target3_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target3'])
    next_days['target4_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target4'])

    if training:
        # Aggregate the cumulative means of the targets, grouped by playerId
        agg = {'date': lambda x: x.values[-1],  # this is for keeping date. it takes the longest time
               'playerId': 'min',
               'target1_med': 'median', 'target2_med': 'median', 'target3_med': 'median', 'target4_med': 'median',
               'target1_mean': 'mean', 'target2_mean': 'mean', 'target3_mean': 'mean', 'target4_mean': 'mean',
               'target1_y_game': 'median', 'target2_y_game': 'median', 'target3_y_game': 'median', 'target4_y_game': 'median',
               'target1_n_game': 'median', 'target2_n_game': 'median', 'target3_n_game': 'median', 'target4_n_game': 'median',
               'target1_y_game_mean': 'mean', 'target2_y_game_mean': 'mean', 'target3_y_game_mean': 'mean', 'target4_y_game_mean': 'mean',
               'target1_n_game_mean': 'mean', 'target2_n_game_mean': 'mean', 'target3_n_game_mean': 'mean', 'target4_n_game_mean': 'mean'
               }
        agg_targets = next_days.groupby(['playerId']).expanding().agg(agg).reset_index(drop=True)

        # Shift the targets down by 1, because we can't use the target on its own day (only from previous days)
        # TODO: instead of fillna(0), use the unknown player's mean/medians
        shifted = agg_targets.drop(['date'], 1).groupby(['playerId']).shift(1).fillna(0)
        print(shifted.columns)
        agg_targets[['target1_med', 'target2_med', 'target3_med', 'target4_med',
                     'target1_mean', 'target2_mean', 'target3_mean', 'target4_mean',
                     'target1_y_game', 'target2_y_game', 'target3_y_game', 'target4_y_game',
                     'target1_n_game', 'target2_n_game', 'target3_n_game', 'target4_n_game',
                     'target1_y_game_mean', 'target2_y_game_mean', 'target3_y_game_mean', 'target4_y_game_mean',
                     'target1_n_game_mean', 'target2_n_game_mean', 'target3_n_game_mean', 'target4_n_game_mean'
                     ]] = shifted
        # Convert the date back to date-time
        agg_targets['date'] = pd.to_datetime(agg_targets['date'])
        # Merge to sort it back in order
        agg_targets = next_days_orig.merge(agg_targets, on=['date', 'playerId'])
        print(next_days.shape)
        print(agg_targets.shape)
        agg_targets = agg_targets.drop(['engagementMetricsDate', 'target1', 'target2', 'target3', 'target4'], 1)
    else:
        # next_days.drop(['date', 'numGames', 'target1', 'target2', 'target3', 'target4'], 1)
        mean_group = next_days[['playerId', 'target1_mean', 'target2_mean', 'target3_mean', 'target4_mean',
                                'target1_y_game_mean', 'target2_y_game_mean', 'target3_y_game_mean', 'target4_y_game_mean',
                                'target1_n_game_mean', 'target2_n_game_mean', 'target3_n_game_mean', 'target4_n_game_mean'
                                ]]
        median_group = next_days[['playerId', 'target1_med', 'target2_med', 'target3_med', 'target4_med',
                                  'target1_y_game', 'target2_y_game', 'target3_y_game', 'target4_y_game',
                                  'target1_n_game', 'target2_n_game', 'target3_n_game', 'target4_n_game'
                                  ]]
        mean_group_merge = mean_group.groupby(['playerId']).mean().reset_index()
        median_group_merge = median_group.groupby(['playerId']).median().reset_index()
        agg_targets = mean_group_merge.merge(median_group_merge, on='playerId')
        # Calculate the unknown player
        mean_all = mean_group.mean()
        mean_all.at['playerId'] = -1
        median_all = median_group.median().drop(labels=['playerId'])
        agg_all = pd.concat([mean_all, median_all], axis=0)
        print('agg_all', agg_all)
        agg_targets = agg_targets.append(agg_all, ignore_index=True)
        # Ensure the correct order
        agg_targets = agg_targets[['playerId', 'target1_med', 'target2_med', 'target3_med', 'target4_med',
                                   'target1_mean', 'target2_mean', 'target3_mean', 'target4_mean',
                                   'target1_y_game', 'target2_y_game', 'target3_y_game', 'target4_y_game',
                                   'target1_n_game', 'target2_n_game', 'target3_n_game', 'target4_n_game',
                                   'target1_y_game_mean', 'target2_y_game_mean', 'target3_y_game_mean', 'target4_y_game_mean',
                                   'target1_n_game_mean', 'target2_n_game_mean', 'target3_n_game_mean', 'target4_n_game_mean'
                                   ]]
        print(agg_targets.shape, agg_targets.columns)
    print(agg_targets.head())
    print(agg_targets.tail())
    agg_targets = reduce_mem_usage(agg_targets)
    save_name = 'mlb-merged-data/pre_train.pkl' if training else 'mlb-merged-data/pre_test.pkl'
    agg_targets.to_pickle(save_name, protocol=4)
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


def compute_pre_date_features():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    player_box = player_box_features(pd.read_pickle('mlb-processed-data/playerBoxScores.pkl'))
    merged = next_days.merge(player_box, on=['date', 'playerId'], how='left')
    merged = merged.fillna(0)  # this will fill player_box as 0 for when there was no game played
    next_days = merged[['date', 'playerId', 'numGames', 'target1', 'target2', 'target3', 'target4']].copy()
    next_days = next_days.rename(columns={'target1': 't1', 'target2': 't2', 'target3': 't3', 'target4': 't4'})
    next_days['ty1'] = np.where(next_days['numGames'] == 0, np.nan, next_days['t1'])
    next_days['ty2'] = np.where(next_days['numGames'] == 0, np.nan, next_days['t2'])
    next_days['ty3'] = np.where(next_days['numGames'] == 0, np.nan, next_days['t3'])
    next_days['ty4'] = np.where(next_days['numGames'] == 0, np.nan, next_days['t4'])
    next_days['tn1'] = np.where(next_days['numGames'] > 0, np.nan, next_days['t1'])
    next_days['tn2'] = np.where(next_days['numGames'] > 0, np.nan, next_days['t2'])
    next_days['tn3'] = np.where(next_days['numGames'] > 0, np.nan, next_days['t3'])
    next_days['tn4'] = np.where(next_days['numGames'] > 0, np.nan, next_days['t4'])

    dates = next_days[['date']].copy().reset_index(drop=True)
    dates['year'] = dates['date'].dt.year
    dates['month'] = dates['date'].dt.month
    # dates['day'] = dates['date'].dt.day
    dates['dayOfWeek'] = dates['date'].dt.dayofweek
    # dates['week'] = dates['date'].dt.isocalendar().week
    dates = dates.drop(['date'], 1)
    next_days = next_days.reset_index(drop=True).join(dates)
    targets = ['t1', 't2', 't3', 't4',
               'ty1', 'ty2', 'ty3', 'ty4',
               'tn1', 'tn2', 'tn3', 'tn4']
    # Years - only use data from first month of season, since that's all we have for 2021
    years = next_days[['year', 'date'] + targets]
    years = years[(years['date'] > '2018-03-29') & (years['date'] < '2018-04-28') |
                  (years['date'] > '2019-03-20') & (years['date'] < '2019-04-19') |
                  (years['date'] > '2020-07-23') & (years['date'] < '2020-08-22') |
                  (years['date'] > '2021-04-01') & (years['date'] < '2021-05-01')].drop('date', 1)
    years = years.groupby(['year']).median()
    years.columns = list(map(lambda name: 'year_' + name, years.columns))
    # Months - don't take 2020 values b/c it was irregular season. no 2021 values b/c they're not complete
    months = next_days[['year', 'month'] + targets]
    months = months[(months['year'] < 2020)].drop('year', 1)
    months = months.groupby(['month']).median()
    months.columns = list(map(lambda name: 'month_' + name, months.columns))
    # Days of week - only during the season
    days_of_week = next_days[['dayOfWeek', 'date'] + targets]
    days_of_week = days_of_week[  # (days_of_week['date'] > '2018-03-29') & (days_of_week['date'] <= '2018-10-01') |
                                (days_of_week['date'] > '2019-03-20') & (days_of_week['date'] < '2019-09-29') |
                                (days_of_week['date'] > '2020-07-23') & (days_of_week['date'] < '2020-09-27') |
                                (days_of_week['date'] > '2021-04-01') & (days_of_week['date'] <= '2021-04-30')]
    days_of_week = days_of_week.drop('date', 1)
    days_of_week = days_of_week.groupby(['dayOfWeek']).median()
    days_of_week.columns = list(map(lambda name: 'day_' + name, days_of_week.columns))
    # Save
    years.to_pickle('mlb-merged-data/years.pkl', protocol=4)
    months.to_pickle('mlb-merged-data/months.pkl', protocol=4)
    days_of_week.to_pickle('mlb-merged-data/days_of_week.pkl', protocol=4)
    print(years)
    print(months)
    print(days_of_week)


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
    merged.to_pickle('mlb-merged-data/merged.pkl', protocol=5)


def main():
    compute_pre_features(False)
    # compute_pre_date_features()
    # Compute the merged features
    # merged = compute_all_merge_features()
    # saved_merged(merged)
    # Test the test flow
    # next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    # player_box = pd.read_pickle('mlb-processed-data/playerBoxScores.pkl')
    # pre_agg = pd.read_pickle('mlb-merged-data/pre_train.pkl')
    # compute_merge_features(next_days, player_box, pre_agg, True)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
