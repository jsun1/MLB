import datetime
import matplotlib.pyplot as plt
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
    players = pd.read_csv('mlb-player-digital-engagement-forecasting/players.csv')
    roster = pd.read_pickle('mlb-processed-data/rosters.pkl')
    roster_agg = pd.read_pickle('mlb-merged-data/status.pkl')
    trans = pd.read_pickle('mlb-processed-data/transactions.pkl')
    trans_agg = pd.read_pickle('mlb-merged-data/pre_transactions.pkl')
    team_box = pd.read_pickle('mlb-processed-data/teamBoxScores.pkl')
    award = pd.read_pickle('mlb-processed-data/awards.pkl')
    award_agg = pd.read_pickle('mlb-merged-data/pre_awards.pkl')
    pre_roll = pd.read_pickle('mlb-merged-data/roll_train.pkl')
    # positions = pd.read_pickle('mlb-merged-data/positions.pkl')
    # print(next_days.head())
    # print(player_box.head())
    # print(next_days.shape, player_box.shape)
    # print(player_box.columns)
    # player_box has 3 unique keys: playerId, date, gamePk
    return compute_merge_features(next_days, player_box, pre_agg, year, month, day, players, roster, roster_agg, trans, trans_agg, team_box, award, award_agg, pre_roll, True)


# Put this in the Notebook!
def compute_merge_features(next_days, player_box, pre_agg, year, month, day, players, roster, roster_agg, trans, trans_agg, team_box, award, award_agg, pre_roll, training):
    if training:
        # Limit training only to the season
        next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
                              (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
                              (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27') |
                              (next_days['date'] >= '2021-04-01')]
        # Add whether they are test set players
        # players = pd.read_csv('mlb-player-digital-engagement-forecasting/players.csv', usecols=['playerId', 'playerForTestSetAndFuturePreds'])
        # players['playerForTestSetAndFuturePreds'] = players['playerForTestSetAndFuturePreds'].apply(lambda x: 1.0 if x == True else 0.0)
        # next_days = next_days.merge(players, on=['playerId'], how='left')
        # next_days = next_days[(next_days['playerForTestSetAndFuturePreds'] == True)].drop(['playerForTestSetAndFuturePreds'], 1)
    # orig_player_box = player_box.copy()
    player_box = player_box_features(player_box)
    # player_box = player_box.merge(positions, on=['positionCode'], how='left')
    merged = next_days.merge(player_box, on=['date', 'playerId'], how='left')
    merged = merged.fillna(0)  # this will fill player_box as 0 for when there was no game played
    # Player features
    # players = player_features(players)
    # merged = merged.merge(players, on=['playerId'], how='left')
    # Team box features
    # team_box = team_box_features(orig_player_box, team_box)
    # merged = merged.merge(team_box, on=['date', 'playerId'], how='left')
    # merged = merged.fillna(0)  # this will fill team_box as 0 for when there was no game played
    # Roster features
    merged = merged.merge(roster[['date', 'playerId', 'statusCode']], on=['date', 'playerId'], how='left')
    merged = merged.merge(roster_agg, on=['statusCode'], how='left')
    merged = merged.drop(['statusCode'], 1).fillna(0)
    # Transactions features
    transactions = trans[['date', 'playerId', 'typeCode']].dropna()#.drop_duplicates(subset=['date', 'playerId'])
    trans_merge = merged.merge(transactions, on=['date', 'playerId'], how='left')[['date', 'playerId', 'typeCode']]
    trans_merge = trans_merge.merge(trans_agg, on=['typeCode'], how='left')
    trans_merge = trans_merge.drop(['typeCode'], 1).fillna(0)
    trans_merge = trans_merge.groupby(by=['date', 'playerId'], sort=False).max()
    merged = merged.merge(trans_merge, on=['date', 'playerId'], how='left')
    # Awards
    award = award[['date', 'playerId', 'awardId']]
    award_merge = merged.merge(award, on=['date', 'playerId'], how='left')[['date', 'playerId', 'awardId']]
    award_merge = award_merge.merge(award_agg, on=['awardId'], how='left')
    award_merge = award_merge.drop(['awardId'], 1).fillna(0)
    award_merge = award_merge.groupby(by=['date', 'playerId'], sort=False).max()
    merged = merged.merge(award_merge, on=['date', 'playerId'], how='left')
    # Twitter followers
    # print(merged.shape)
    # twitter = pd.read_pickle('mlb-processed-data/playerTwitterFollowers.pkl')
    # print(twitter.tail())
    # twitter = twitter[['date', 'playerId', 'numberOfFollowers']]
    # twitter_median = twitter['numberOfFollowers'].median()
    # merged = merged.merge(twitter, on=['date', 'playerId'], how='left')
    # merged['numberOfFollowers'] = merged['numberOfFollowers'].fillna(twitter_median)
    # print(twitter_median, twitter, merged.shape)
    # Date features
    merged = merge_date_features(merged, year, month, day, training)
    merged = merge_pre_agg(merged, pre_agg, training)
    merged = merge_pre_roll(merged, pre_roll, training)
    # These features have low activation
    # merged = merged.drop(['caughtStealing', 'groundOutsPitching', 'hitByPitch', 'day_t3', 'wildPitches', 'rbi',
    #                       'numGames', 'blownSaves', 'doubles', 'catchersInterferencePitching'], 1)
                          #                                      'runsScored', 'triples',
                          # 'caughtStealingPitching', 'rbiPitching', 'assists', 'groundIntoDoublePlay', 'day_tg4',
                          # 'plateAppearances', 'atBatsPitching', 'groundIntoTriplePlay'], 1)
    merged = merge_real_offsets(merged, training)
    # Dataset normalization
    not_feature = ['date', 'playerId', 'target1', 'target2', 'target3', 'target4']
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
    merged[list(norm.columns)] = norm
    if not training:
        merged = merged.fillna(0)  # Just in case there are unknown player ids
    # print(merged.isna().sum().to_string())
    # merged = reduce_mem_usage(merged)
    return merged


# Put this in the Notebook!
def merge_offsets(merged, training):
    if training:
        index = merged[['date', 'playerId']].sort_values(['playerId', 'date']).reset_index(drop=True)
        targets = merged[['date', 'playerId', 'target1', 'target2', 'target3', 'target4']]
        targets = targets.sort_values(['playerId', 'date']).reset_index(drop=True)
        targets = targets.drop('date', 1)
        targets_1d = targets.groupby(['playerId']).shift(1).fillna(0)
        index[['1d_t1', '1d_t2', '1d_t3', '1d_t4']] = targets_1d
        # targets_2d = targets.groupby(['playerId']).shift(2).fillna(0)
        # index[['2d_t1', '2d_t2', '2d_t3', '2d_t4']] = targets_2d
        merged = merged.merge(index, on=['date', 'playerId'], how='left')
    else:
        pass
    return merged


# Put this in the Notebook!
def merge_real_offsets(merged, training):
    global OFFSETS1
    global OFFSETS2
    global OFFSETS3
    if training:
        index = merged[['date', 'playerId']].sort_values(['playerId', 'date']).reset_index(drop=True)
        # targets = merged[['date', 'playerId', 'target1', 'target2', 'target3', 'target4']]
        targets = merged.drop(['engagementMetricsDate', 'target1', 'target2', 'target3', 'target4',
                               'target1_med', 'target2_med', 'target3_med', 'target4_med',
                               'target1_mean', 'target2_mean', 'target3_mean', 'target4_mean'], 1)#,
                               # 'playerName', 'DOB', 'mlbDebutDate', 'heightInches', 'weight', 'primaryPositionCode',
                               # 'playerForTestSetAndFuturePreds', 'birthCountryUSA', 'birthCountryDR', 'birthCountryOther'], 1)
        for target_col in targets.columns:
            if target_col.startswith('team_') or target_col.startswith('year_') or target_col.startswith('roll'):
                targets = targets.drop([target_col], 1)

        targets = targets.sort_values(['playerId', 'date']).reset_index(drop=True)
        targets = targets.drop('date', 1)
        targets_1d_orig = targets.rename(mapper=lambda x: x if x == 'playerId' else '1d_' + x, axis=1)
        targets_1d = targets_1d_orig.groupby(['playerId']).shift(1).fillna(0)
        index[list(targets_1d.columns)] = targets_1d
        targets_2d_orig = targets.rename(mapper=lambda x: x if x == 'playerId' else '2d_' + x, axis=1)
        targets_2d = targets_2d_orig.groupby(['playerId']).shift(2).fillna(0)
        index[list(targets_2d.columns)] = targets_2d
        targets_3d_orig = targets.rename(mapper=lambda x: x if x == 'playerId' else '3d_' + x, axis=1)
        targets_3d = targets_3d_orig.groupby(['playerId']).shift(3).fillna(0)
        index[list(targets_3d.columns)] = targets_3d
        # targets_2d = targets.groupby(['playerId']).shift(2).fillna(0)
        # index[['2d_t1', '2d_t2', '2d_t3', '2d_t4']] = targets_2d
        merged = merged.merge(index, on=['date', 'playerId'], how='left')
        # Compute and save the last values for testing
        to_save = targets_1d_orig.groupby(['playerId']).last().fillna(0)
        to_save.to_pickle('mlb-merged-data/1d.pkl', protocol=4)
        to_save = targets_2d_orig.groupby(['playerId']).nth(-2).fillna(0)
        to_save.to_pickle('mlb-merged-data/2d.pkl', protocol=4)
        to_save = targets_3d_orig.groupby(['playerId']).nth(-3).fillna(0)
        to_save.to_pickle('mlb-merged-data/3d.pkl', protocol=4)
    else:
        # Update the offsets
        to_update = merged.drop(
            ['target1', 'target2', 'target3', 'target4', 'target1_med', 'target2_med', 'target3_med', 'target4_med',
             'target1_mean', 'target2_mean', 'target3_mean', 'target4_mean'], 1)
             #'playerName', 'DOB', 'mlbDebutDate', 'heightInches', 'weight', 'primaryPositionCode',
              #                 'playerForTestSetAndFuturePreds', 'birthCountryUSA', 'birthCountryDR', 'birthCountryOther'], 1)
        for target_col in to_update.columns:
            if target_col.startswith('team_') or target_col.startswith('year_') or target_col.startswith('roll'):
                to_update = to_update.drop([target_col], 1)
        to_update = to_update.drop('date', 1)
        to_update = to_update.rename(mapper=lambda x: x if x == 'playerId' else '1d_' + x, axis=1)
        to_update = to_update.groupby(['playerId']).last()
        # Merge in the offsets based on playerId
        merged = merged.merge(OFFSETS1, on=['playerId'], how='left').fillna(0)
        merged = merged.merge(OFFSETS2, on=['playerId'], how='left').fillna(0)
        merged = merged.merge(OFFSETS3, on=['playerId'], how='left').fillna(0)
        # Actually update offsets
        OFFSETS3 = OFFSETS2.rename(mapper=lambda x: x if x == 'playerId' else x.replace("2d_", "3d_"), axis=1)
        OFFSETS2 = OFFSETS1.rename(mapper=lambda x: x if x == 'playerId' else x.replace("1d_", "2d_"), axis=1)
        OFFSETS1 = pd.concat([OFFSETS1, to_update]).groupby(['playerId']).last()
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
    # for name in ['month', 'day']:
        for i in ['1', '2', '3', '4']:
            to_drop.append(name + '_ty' + i)
            to_drop.append(name + '_tn' + i)
    merged = merged.drop(to_drop, 1)
    if not training:
        merged = merged.fillna(0)  # the test code has bad dates
    return merged


# Add linear date features (they didn't help with the training/validation loss)
def merge_date_features_linear(merged, next_days):
    dates = next_days[['date']].copy().reset_index(drop=True)
    # dates['year'] = dates['date'].dt.year.apply(lambda x: 0 if x <= 2019 else 1)
    # dates['month'] = dates['date'].dt.month
    # dates['day'] = dates['date'].dt.day
    dates['dayOfWeek'] = dates['date'].dt.dayofweek
    # dates['week'] = dates['date'].dt.isocalendar().week
    dates['dayOfYear'] = dates['date'].dt.dayofyear
    dates = dates.drop(['date'], 1)
    # merged[['year', 'month', 'day', 'dayOfWeek', 'week', 'dayOfYear']] = dates
    merged[['dayOfWeek', 'dayOfYear']] = dates
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

    player_box = player_box.drop(['gamePk', 'gameDate', 'teamId', 'jerseyNum',
       'teamName', 'playerName', 'positionName', 'positionType'], 1).reset_index(drop=True)
    # Convert game time to float
    player_box['gameTimeUTC'] = pd.to_datetime(player_box['gameTimeUTC']).dt.time
    player_box['gameTimeUTC'] = player_box['gameTimeUTC'].apply(lambda x: x.hour + x.minute / 60)
    # player_box['gameTimeUTC_min'] = player_box['gameTimeUTC']
    # player_box['gameTimeUTC_max'] = player_box['gameTimeUTC']
    # Convert jerseyNum to int
    # a = player_box[['jerseyNum']]
    # for tuple in a.itertuples(index=True):
    #     t = type(tuple[1])
    #     if t != type('33') and t != type(0) and t != type(0.1):
    #         print('type', t)
    #     if t == type('ww'):
    #         print('mee', tuple[1])
    # print(player_box[['jerseyNum']].shape)
    # print('isna', player_box[['jerseyNum']].isna().sum())
    # print(player_box[['home']]).tail()
    # print(player_box[['jerseyNum']].info(verbose=True))
    # player_box['jerseyNum'] = player_box['jerseyNum'].apply(lambda x: 0 if (x is None or x == '' or (not isinstance(x, str) and np.isnan(x))) else int(x))

    # Add number of games played
    player_box['numGames'] = 1
    player_box['hasGame'] = 1

    first = lambda x: x.iloc[0]
    num_columns = len(player_box.columns)
    style = 'max'
    player_box = player_box.groupby(['date', 'playerId']).agg({'numGames': 'sum', 'hasGame': 'max',
        'home': 'max', #'gamePk': first, 'gameDate': first,
        'gameTimeUTC': 'median', #'gameTimeUTC_min': 'min', 'gameTimeUTC_max': 'max',  #'teamId': first,
       #'teamName': first, 'playerName': first,
        #'jerseyNum': 'min',
        'positionCode': first,
       #'positionName': first, 'positionType': first,
        'battingOrder': 'min', 'gamesPlayedBatting': style,
       'flyOuts': style, 'groundOuts': style, 'runsScored': style, 'doubles': style, 'triples': style, 'homeRuns': style,
       'strikeOuts': style, 'baseOnBalls': style, 'intentionalWalks': style, 'hits': style, 'hitByPitch': style,
       'atBats': style, 'caughtStealing': style, 'stolenBases': style, 'groundIntoDoublePlay': style,
       'groundIntoTriplePlay': style, 'plateAppearances': style, 'totalBases': style, 'rbi': style,
       'leftOnBase': style, 'sacBunts': style, 'sacFlies': style, 'catchersInterference': style,
       'pickoffs': style, 'gamesPlayedPitching': style, 'gamesStartedPitching': style,
       'completeGamesPitching': style, 'shutoutsPitching': style, 'winsPitching': style,
       'lossesPitching': style, 'flyOutsPitching': style, 'airOutsPitching': style,
       'groundOutsPitching': style, 'runsPitching': style, 'doublesPitching': style,
       'triplesPitching': style, 'homeRunsPitching': style, 'strikeOutsPitching': style,
       'baseOnBallsPitching': style, 'intentionalWalksPitching': style, 'hitsPitching': style,
       'hitByPitchPitching': style, 'atBatsPitching': style, 'caughtStealingPitching': style,
       'stolenBasesPitching': style, 'inningsPitched': style, 'saveOpportunities': style,  # inningsPitched is a float (0.1)
       'earnedRuns': style, 'battersFaced': style, 'outsPitching': style, 'pitchesThrown': style, 'balls': style,
       'strikes': style, 'hitBatsmen': style, 'balks': style, 'wildPitches': style, 'pickoffsPitching': style,
       'rbiPitching': style, 'gamesFinishedPitching': style, 'inheritedRunners': style,
       'inheritedRunnersScored': style, 'catchersInterferencePitching': style,
       'sacBuntsPitching': style, 'sacFliesPitching': style, 'saves': style, 'holds': style, 'blownSaves': style,
       'assists': style, 'putOuts': style, 'errors': style, 'chances': style}).reset_index()
    assert(len(player_box.columns) == num_columns)
    # Add a column for total number of players that have games on a given day (not helpful)
    # total = player_box[['date', 'numGames']].groupby(['date']).sum().rename(columns={'numGames': 'totalPlayerGames'})
    # player_box = player_box.merge(total, on='date', how='left')
    return player_box


def team_box_features(player_box, team_box):
    # player_box = player_box[['playerId', 'date', 'gamePk', 'teamId']]
    team_box = team_box.drop(['home', 'gameDate', 'gameTimeUTC'], 1)
    # Drop these because they're all zero
    team_box = team_box.drop(['groundIntoTriplePlay', 'airOutsPitching', 'groundOutsPitching', 'doublesPitching',
                              'triplesPitching', 'balks', 'wildPitches', 'inheritedRunners', 'inheritedRunnersScored'], 1)
    # print(team_box.columns)
    team_box = team_box.rename(mapper=lambda x: x if x in ['date', 'teamId', 'gamePk'] else 'team_' + x, axis=1)
    player_box = player_box.reset_index(drop=True)
    merged = player_box[['playerId', 'date', 'gamePk', 'teamId']].merge(team_box, on=['date', 'teamId', 'gamePk'], how='left')
    # Sum up all team_box for the day
    # team_box_sum = team_box.drop(['teamId', 'gamePk'], 1).groupby(['date']).sum()
    # merged = player_box[['playerId', 'date']].merge(team_box_sum, on=['date'], how='left')
    # Get max of player box
    # player_box_max = player_box.drop(
    #     ['playerId', 'groundIntoTriplePlay', 'airOutsPitching', 'groundOutsPitching', 'doublesPitching',
    #      'triplesPitching', 'balks', 'wildPitches', 'inheritedRunners', 'inheritedRunnersScored'],
    #     1)
    # player_box_max = player_box_max[['date', 'flyOuts', 'groundOuts', 'runsScored',
    #    'doubles', 'triples', 'homeRuns', 'strikeOuts', 'baseOnBalls',
    #    'intentionalWalks', 'hits', 'hitByPitch', 'atBats', 'caughtStealing',
    #    'stolenBases', 'groundIntoDoublePlay', 'plateAppearances', 'totalBases',
    #    'rbi', 'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
    #    'pickoffs', 'runsPitching', 'homeRunsPitching', 'strikeOutsPitching',
    #    'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
    #    'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
    #    'stolenBasesPitching', 'inningsPitched', 'earnedRuns', 'battersFaced',
    #    'outsPitching', 'hitBatsmen', 'pickoffsPitching', 'rbiPitching',
    #    'catchersInterferencePitching', 'sacBuntsPitching', 'sacFliesPitching']]
    # player_box_max = player_box_max.groupby('date').max()
    # player_box_max = player_box_max.rename(mapper=lambda x: x if x in ['date', 'playerId'] else 'team_' + x, axis=1)
    # merged = player_box[['playerId', 'date']].merge(player_box_max, on=['date'], how='left')
    # Divide each player box by the team box
    assert(merged.shape[0] == player_box.shape[0])
    player_box = player_box.fillna(0)
    assert(merged['playerId'].equals(player_box['playerId']))
    for col in team_box.columns:
        if col not in ['date', 'teamId', 'gamePk']:
            assert(col.startswith('team_'))
            player_col = col[len('team_'):]
            merged[col] = player_box[player_col] / merged[col]
    # merged = merged.fillna(0)  # this is because there was division by 0
    merged = merged.replace(np.inf, 0)

    # Get information from the other team
    # team_box_min = team_box[['teamId', 'gamePk']].groupby(['gamePk']).min().rename(mapper=lambda x: 'min_teamId' if x == 'teamId' else x, axis=1)
    # team_box_max = team_box[['teamId', 'gamePk']].groupby(['gamePk']).max().rename(mapper=lambda x: 'max_teamId' if x == 'teamId' else x, axis=1)
    # team_box_other = team_box[['date', 'teamId', 'gamePk']]
    # team_box_other = team_box_other.merge(team_box_min, on=['gamePk'], how='left')
    # team_box_other = team_box_other.merge(team_box_max, on=['gamePk'], how='left')
    # team_box_other['other_teamId'] = np.where(team_box_other['teamId'] == team_box_other['min_teamId'], team_box_other['max_teamId'], team_box_other['min_teamId'])
    # team_box_other_team = team_box.rename(mapper=lambda x: x if x in ['date', 'gamePk'] else 'other_' + x, axis=1)#.set_index('other_teamId')
    # team_box_other = team_box_other.merge(team_box, on=['date', 'gamePk', 'teamId'], how='left')
    # team_box_other = team_box_other.merge(team_box_other_team, on=['date', 'gamePk', 'other_teamId'], how='left')
    # # team_box_other['team_won'] = np.where(team_box_other['team_runsScored'] > team_box_other['other_team_runsScored'], 1, 0)
    # # team_box_other['team_won'] = np.where(team_box_other['team_runsScored'] == team_box_other['other_team_runsScored'], 0.5, team_box_other['team_won'])
    # team_box_other['team_won'] = team_box_other['team_runsScored'] - team_box_other['other_team_runsScored']
    # team_box_other = team_box_other[['date', 'teamId', 'gamePk', 'team_won']]
    # print(team_box_other.head())
    # print(team_box_other.tail())
    # merged = player_box[['playerId', 'date', 'gamePk', 'teamId']].merge(team_box_other, on=['date', 'teamId', 'gamePk'], how='left')
    # Aggregate
    team_agg = merged.drop(['gamePk', 'teamId'], 1)
    # team_agg = merged
    team_agg = team_agg.groupby(['date', 'playerId']).median()#.sum()
    return team_agg


# Put this in the Notebook!
def player_features(players):
    players = players.drop(['birthCity', 'birthStateProvince', 'primaryPositionName'], 1)
    """
    # Use length of player name
    players['playerName'] = players['playerName'].apply(lambda x: len(x))
    # Date of birth
    players['DOB'] = pd.to_numeric(pd.to_datetime(players['DOB']))
    # MLB debut date (fill with median date)
    players['mlbDebutDate'] = pd.to_datetime(players['mlbDebutDate']).fillna(value=pd.to_datetime('2017-04-03'))
    players['mlbDebutDate'] = pd.to_numeric(players['mlbDebutDate'])
    # Birth country - one hot
    players['birthCountryUSA'] = np.where(players['birthCountry'] == 'USA', 1, 0)
    players['birthCountryDR'] = np.where(players['birthCountry'] == 'Dominican Republic', 1, 0)
    players['birthCountryOther'] = np.where((players['birthCountry'] != 'USA') & (players['birthCountry'] != 'Dominican Republic'), 1, 0)
    players = players.drop(['birthCountry'], 1)
    # Height and weight
    # Primary position code
    players['primaryPositionCode'] = players['primaryPositionCode'].apply(lambda x: int(x) if x.isdigit() else 0)
    """
    # Use 0/1 for whether in test set
    players['playerForTestSetAndFuturePreds'] = players['playerForTestSetAndFuturePreds'].apply(lambda x: 1.0 if x == True else 0.0)
    players = players[['playerId', 'playerForTestSetAndFuturePreds']]
    return players


# Put this in the Notebook!
def merge_pre_agg(merged, pre_agg, training):
    # From the pre_train/test.pkl aggregations, compute just the final values
    assert(merged['numGames'].isna().sum() == 0)
    m_shape = merged.shape
    if training:
        # In training, use all values
        merged = merged.merge(pre_agg, on=['date', 'playerId'])
    else:
        lookup = pre_agg
        # Assume pre_agg is a lookup data frame which can be merged via playerId
        merged = merged.merge(lookup, on=['playerId'], how='left')
        # Account for unknown players
        unknown = pre_agg[(pre_agg['playerId'] == -1)].reset_index(drop=True).copy().iloc[0]
        merged = merged.fillna(value=unknown)
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
    assert(merged.isna().sum().sum() == 0)
    return merged


# Put this in the Notebook!
def merge_pre_roll(merged, pre_roll, training):
    # From the pre_train/test.pkl aggregations, compute just the final values
    assert(merged['numGames'].isna().sum() == 0)
    m_shape = merged.shape
    if training:
        # In training, use all values
        # print('pre roll', pre_roll.shape)
        # test_roll = pd.read_pickle('mlb-merged-data/roll_test_45.pkl')
        # pre_roll_test = pre_roll[(pre_roll['date'] >= '2021-06-10')]
        # pre_roll_test = pre_roll_test.merge(test_roll, on=['date', 'playerId'], how='left', suffixes=['_x', None])
        # to_drop = []
        # for col in pre_roll_test.columns:
        #     if col.endswith('_x'):
        #         to_drop.append(col)
        # pre_roll_test = pre_roll_test.drop(to_drop, 1)
        # # Account for unknown players
        # unknown = test_roll[(test_roll['playerId'] == -1) & (test_roll['date'] == '2021-06-10')].reset_index(drop=True).copy().iloc[0]
        # pre_roll_test = pre_roll_test.fillna(value=unknown)
        # pre_roll = pre_roll[(pre_roll['date'] < '2021-06-10')]
        # print('pre roll', pre_roll.shape)
        # pre_roll = pd.concat([pre_roll, pre_roll_test], ignore_index=True)
        # print('pre roll', pre_roll.shape)
        merged = merged.merge(pre_roll, on=['date', 'playerId'])
    else:
        lookup = pre_roll
        # Assume pre_roll is a lookup data frame which can be merged via playerId
        merged = merged.merge(lookup, on=['playerId'], how='left')
        # Account for unknown players
        unknown = pre_roll[(pre_roll['playerId'] == -1)].reset_index(drop=True).copy().iloc[0]
        merged = merged.fillna(value=unknown)
    # Use the yes/no game target depending on numGames
    merged['roll1_game'] = np.where(merged['numGames'] == 0, merged['roll1_n_game'], merged['roll1_y_game'])
    merged['roll2_game'] = np.where(merged['numGames'] == 0, merged['roll2_n_game'], merged['roll2_y_game'])
    merged['roll3_game'] = np.where(merged['numGames'] == 0, merged['roll3_n_game'], merged['roll3_y_game'])
    merged['roll4_game'] = np.where(merged['numGames'] == 0, merged['roll4_n_game'], merged['roll4_y_game'])
    merged['roll1_game_mean'] = np.where(merged['numGames'] == 0, merged['roll1_n_game_mean'], merged['roll1_y_game_mean'])
    merged['roll2_game_mean'] = np.where(merged['numGames'] == 0, merged['roll2_n_game_mean'], merged['roll2_y_game_mean'])
    merged['roll3_game_mean'] = np.where(merged['numGames'] == 0, merged['roll3_n_game_mean'], merged['roll3_y_game_mean'])
    merged['roll4_game_mean'] = np.where(merged['numGames'] == 0, merged['roll4_n_game_mean'], merged['roll4_y_game_mean'])
    merged = merged.drop(['roll1_y_game', 'roll2_y_game', 'roll3_y_game', 'roll4_y_game',
                          'roll1_n_game', 'roll2_n_game', 'roll3_n_game', 'roll4_n_game',
                          'roll1_y_game_mean', 'roll2_y_game_mean', 'roll3_y_game_mean', 'roll4_y_game_mean',
                          'roll1_n_game_mean', 'roll2_n_game_mean', 'roll3_n_game_mean', 'roll4_n_game_mean'
                          ], 1)
    # Drop the means
    # merged = merged.drop(['roll1_mean', 'roll1_mean', 'roll1_mean', 'roll1_mean',
    #                       'roll1_game_mean', 'roll2_game_mean', 'roll3_game_mean', 'roll4_game_mean'], 1)
    assert(merged.shape[0] == m_shape[0])
    assert(merged.isna().sum().sum() == 0)
    return merged


# Computes features that we pre-compute and load into the notebook pre-inference
def compute_pre_features(training=True):
    next_days_orig = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    # Limit computation to only the season and the preceding month
    # next_days_orig = next_days_orig[(next_days_orig['date'] >= '2018-02-28') & (next_days_orig['date'] <= '2018-10-01') |
    #                                 (next_days_orig['date'] >= '2019-02-20') & (next_days_orig['date'] <= '2019-09-29') |
    #                                 (next_days_orig['date'] >= '2020-06-23') & (next_days_orig['date'] <= '2020-09-27') |
    #                                 (next_days_orig['date'] >= '2021-03-01') & (next_days_orig['date'] <= '2021-04-30')]
    # if not training:
    #     next_days_orig = next_days_orig[(next_days_orig['date'] >= '2019-05-01')]
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
        # print(agg_targets.head())
        # TODO: Compute the last day's target (for lag)

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


# Computes features that we pre-compute and load into the notebook pre-inference
def compute_pre_rolling(training=True):
    next_days_orig = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    # Limit computation to only the season
    # next_days_orig = next_days_orig[(next_days_orig['date'] >= '2018-03-28') & (next_days_orig['date'] <= '2018-10-01') |
    #                                 (next_days_orig['date'] >= '2019-03-20') & (next_days_orig['date'] <= '2019-09-29') |
    #                                 (next_days_orig['date'] >= '2020-07-23') & (next_days_orig['date'] <= '2020-09-27') |
    #                                 (next_days_orig['date'] >= '2021-04-01')]
    if not training:
        next_days_orig = next_days_orig[(next_days_orig['date'] >= '2021-06-03')]  # 45 days before 7/17/2021
        # next_days_orig = next_days_orig[(next_days_orig['date'] >= '2021-03-27') & (next_days_orig['date'] < '2021-06-10')]  # 75 days before 6/10/2021
    player_box = player_box_features(pd.read_pickle('mlb-processed-data/playerBoxScores.pkl'))
    merged = next_days_orig.merge(player_box, on=['date', 'playerId'], how='left')
    merged = merged.fillna(0)  # this will fill player_box as 0 for when there was no game played
    next_days = merged[['date', 'playerId', 'numGames', 'target1', 'target2', 'target3', 'target4']].copy()
    # next_days_use = next_days_use[next_days_use['date'] < '2018-06-01']
    # Convert the date to numeric so it can be passed into group-by
    next_days['date'] = pd.to_numeric(next_days['date'])  # next_days = next_days.assign(date=pd.to_numeric(next_days['date']))
    next_days[['roll1_med', 'roll2_med', 'roll3_med', 'roll4_med']] = next_days[['target1', 'target2', 'target3', 'target4']]
    next_days[['roll1_mean', 'roll2_mean', 'roll3_mean', 'roll4_mean']] = next_days[['target1', 'target2', 'target3', 'target4']]
    next_days['roll1_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target1'])
    next_days['roll2_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target2'])
    next_days['roll3_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target3'])
    next_days['roll4_y_game'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target4'])
    next_days['roll1_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target1'])
    next_days['roll2_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target2'])
    next_days['roll3_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target3'])
    next_days['roll4_n_game'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target4'])
    next_days['roll1_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target1'])
    next_days['roll2_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target2'])
    next_days['roll3_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target3'])
    next_days['roll4_y_game_mean'] = np.where(next_days['numGames'] == 0, np.nan, next_days['target4'])
    next_days['roll1_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target1'])
    next_days['roll2_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target2'])
    next_days['roll3_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target3'])
    next_days['roll4_n_game_mean'] = np.where(next_days['numGames'] > 0, np.nan, next_days['target4'])

    if training:
        # Aggregate the cumulative means of the targets, grouped by playerId
        agg = {'date': lambda x: x.values[-1],  # this is for keeping date. it takes the longest time
               'playerId': 'min',
               'roll1_med': 'median', 'roll2_med': 'median', 'roll3_med': 'median', 'roll4_med': 'median',
               'roll1_mean': 'mean', 'roll2_mean': 'mean', 'roll3_mean': 'mean', 'roll4_mean': 'mean',
               'roll1_y_game': 'median', 'roll2_y_game': 'median', 'roll3_y_game': 'median', 'roll4_y_game': 'median',
               'roll1_n_game': 'median', 'roll2_n_game': 'median', 'roll3_n_game': 'median', 'roll4_n_game': 'median',
               'roll1_y_game_mean': 'mean', 'roll2_y_game_mean': 'mean', 'roll3_y_game_mean': 'mean', 'roll4_y_game_mean': 'mean',
               'roll1_n_game_mean': 'mean', 'roll2_n_game_mean': 'mean', 'roll3_n_game_mean': 'mean', 'roll4_n_game_mean': 'mean'
               }
        agg_targets = next_days.groupby(['playerId']).rolling(45, min_periods=1).agg(agg).reset_index(drop=True)

        # Shift the targets down by 1, because we can't use the target on its own day (only from previous days)
        # TODO: instead of fillna(0), use the unknown player's mean/medians
        shifted = agg_targets.drop(['date'], 1).groupby(['playerId']).shift(1).fillna(0)
        print(shifted.columns)
        agg_targets[['roll1_med', 'roll2_med', 'roll3_med', 'roll4_med',
                     'roll1_mean', 'roll2_mean', 'roll3_mean', 'roll4_mean',
                     'roll1_y_game', 'roll2_y_game', 'roll3_y_game', 'roll4_y_game',
                     'roll1_n_game', 'roll2_n_game', 'roll3_n_game', 'roll4_n_game',
                     'roll1_y_game_mean', 'roll2_y_game_mean', 'roll3_y_game_mean', 'roll4_y_game_mean',
                     'roll1_n_game_mean', 'roll2_n_game_mean', 'roll3_n_game_mean', 'roll4_n_game_mean'
                     ]] = shifted
        # Convert the date back to date-time
        agg_targets['date'] = pd.to_datetime(agg_targets['date'])
        # Merge to sort it back in order
        agg_targets = next_days_orig.merge(agg_targets, on=['date', 'playerId'])
        print(next_days.shape)
        print(agg_targets.shape)
        agg_targets = agg_targets.drop(['engagementMetricsDate', 'target1', 'target2', 'target3', 'target4'], 1)
        # For training, freeze after 06-01

    else:
        mean_group = next_days[['playerId', 'roll1_mean', 'roll2_mean', 'roll3_mean', 'roll4_mean',
                                'roll1_y_game_mean', 'roll2_y_game_mean', 'roll3_y_game_mean', 'roll4_y_game_mean',
                                'roll1_n_game_mean', 'roll2_n_game_mean', 'roll3_n_game_mean', 'roll4_n_game_mean'
                                ]]
        median_group = next_days[['playerId', 'roll1_med', 'roll2_med', 'roll3_med', 'roll4_med',
                                  'roll1_y_game', 'roll2_y_game', 'roll3_y_game', 'roll4_y_game',
                                  'roll1_n_game', 'roll2_n_game', 'roll3_n_game', 'roll4_n_game'
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
        # print(agg_targets.head())
        # Ensure the correct order
        agg_targets = agg_targets[['playerId', 'roll1_med', 'roll2_med', 'roll3_med', 'roll4_med',
                                   'roll1_mean', 'roll2_mean', 'roll3_mean', 'roll4_mean',
                                   'roll1_y_game', 'roll2_y_game', 'roll3_y_game', 'roll4_y_game',
                                   'roll1_n_game', 'roll2_n_game', 'roll3_n_game', 'roll4_n_game',
                                   'roll1_y_game_mean', 'roll2_y_game_mean', 'roll3_y_game_mean', 'roll4_y_game_mean',
                                   'roll1_n_game_mean', 'roll2_n_game_mean', 'roll3_n_game_mean', 'roll4_n_game_mean'
                                   ]]
        print(agg_targets.shape, agg_targets.columns)
        # Synthesize the dates
        # print('synth', agg_targets.shape)
        # all_together = []
        # start_date = datetime.date(2021, 6, 10)
        # end_date = datetime.date(2021, 7, 17)
        # delta = datetime.timedelta(days=1)
        # while start_date <= end_date:
        #     agg_targets_date = agg_targets.copy()
        #     agg_targets_date['date'] = pd.to_datetime(start_date)
        #     all_together.append(agg_targets_date)
        #     start_date += delta
        # agg_targets = pd.concat(all_together, ignore_index=True)
        # print('synth', agg_targets.shape)
        # print(agg_targets['date'])
    print(agg_targets.head())
    print(agg_targets.tail())
    agg_targets = reduce_mem_usage(agg_targets)
    save_name = 'mlb-merged-data/roll_train.pkl' if training else 'mlb-merged-data/roll_test.pkl'
    agg_targets.to_pickle(save_name, protocol=4)
    return agg_targets


def compute_pre_date_features(training=True):
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
    if training:
        years = years[(years['date'] > '2018-03-29') & (years['date'] < '2018-04-28') |
                      (years['date'] > '2019-03-20') & (years['date'] < '2019-04-19') |
                      (years['date'] > '2020-07-23') & (years['date'] < '2020-08-22') |
                      (years['date'] > '2021-04-01') & (years['date'] < '2021-06-10')].drop('date', 1)
    else:
        years = years[(years['date'] > '2018-03-29') & (years['date'] < '2018-04-28') |
                      (years['date'] > '2019-03-20') & (years['date'] < '2019-04-19') |
                      (years['date'] > '2020-07-23') & (years['date'] < '2020-08-22') |
                      (years['date'] > '2021-04-01')].drop('date', 1)
    years = years.groupby(['year']).median()
    years.columns = list(map(lambda name: 'year_' + name, years.columns))
    # Months - don't take 2020 values b/c it was irregular season. no 2021 values b/c they're not complete
    months = next_days[['year', 'month'] + targets]
    months = months[(months['year'] < 2020)].drop('year', 1)
    months = months.groupby(['month']).median()
    months.columns = list(map(lambda name: 'month_' + name, months.columns))
    # Days of week - only during the season
    days_of_week = next_days[['dayOfWeek', 'date'] + targets]
    if training:
        days_of_week = days_of_week[(days_of_week['date'] >= '2018-03-29') & (days_of_week['date'] <= '2018-10-01') |
                                    (days_of_week['date'] >= '2019-03-20') & (days_of_week['date'] <= '2019-09-29') |
                                    (days_of_week['date'] >= '2020-07-23') & (days_of_week['date'] <= '2020-09-27')]
    else:
        days_of_week = days_of_week[(days_of_week['date'] >= '2018-03-29') & (days_of_week['date'] <= '2018-10-01') |
                                    (days_of_week['date'] >= '2019-03-20') & (days_of_week['date'] <= '2019-09-29') |
                                    (days_of_week['date'] >= '2020-07-23') & (days_of_week['date'] <= '2020-09-27') |
                                    (days_of_week['date'] >= '2021-04-01')]
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


# Computes the median target for each team when they have a game
def compute_pre_team():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
                          (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
                          (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27') |
                          (next_days['date'] >= '2021-04-01')]
    next_days = next_days[['date', 'playerId', 'target1', 'target2', 'target3', 'target4']]
    """
    player_box = pd.read_pickle('mlb-processed-data/playerBoxScores.pkl')
    player_box = player_box[(player_box['date'] >= '2018-03-29') & (player_box['date'] <= '2018-10-01') |
                          (player_box['date'] >= '2019-03-20') & (player_box['date'] <= '2019-09-29') |
                          (player_box['date'] >= '2020-07-23') & (player_box['date'] <= '2020-09-27')]
    print(player_box.shape)
    print(player_box['teamId'].nunique())
    player_box = player_box[['date', 'playerId', 'teamId']]
    player_box.groupby(['date', 'playerId']).min()  # A player may have played > 1 game on a day, so take min teamId
    merged = next_days.merge(player_box, on=['date', 'playerId'], how='left').reset_index(drop=True)
    merged = merged[['teamId', 'target1', 'target2', 'target3', 'target4']]
    # merged = merged.fillna(0)  # this will fill player_box as 0 for when there was no game played
    # merged_y_game = np.where(np.isnan(merged['teamId']), merged['target1_n_game'], merged['target1_y_game'])
    merged_y_game = merged[(merged['teamId'] != np.nan)]
    merged_n_game = merged[(merged['teamId'] == np.nan)]
    teams_y_game = merged_y_game.groupby(['teamId']).median()
    # teams_n_game = merged_n_game.groupby(['teamId']).median()

    # print(teams.shape)
    # print(teams)
    # teams.to_pickle('mlb-merged-data/teams.pkl', protocol=4)

    # print(next_days[(next_days['date'] == '2018-03-29')].shape())
    """

    rosters = pd.read_pickle('mlb-processed-data/rosters.pkl')
    rosters = rosters[(rosters['date'] >= '2018-03-29') & (rosters['date'] <= '2018-10-01') |
                      (rosters['date'] >= '2019-03-20') & (rosters['date'] <= '2019-09-29') |
                      (rosters['date'] >= '2020-07-23') & (rosters['date'] <= '2020-09-27') |
                      (rosters['date'] >= '2021-04-01')]

    # next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
    #                       (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
    #                       (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27') |
    #                       (next_days['date'] >= '2021-04-01') & (next_days['date'] <= '2021-04-30')]
    print('ROSTER')
    print(rosters[['teamId', 'statusCode']].nunique())
    print(rosters.shape)
    print(rosters.columns)
    print(rosters.head())
    print(rosters.tail())
    print(next_days.shape)
    merged = next_days.merge(rosters, on=['date', 'playerId'], how='left').reset_index(drop=True)
    print('m', merged.shape)
    print(merged.head())
    print(merged.tail())
    print(merged['gameDate'].isna().sum())
    print(merged['status'].value_counts())
    print(merged['statusCode'].value_counts())
    status_agg = merged[['statusCode', 'target1', 'target2', 'target3', 'target4']].groupby(['statusCode']).median()
    status_agg = status_agg.rename(columns={'target1': 'status_t1', 'target2': 'status_t2', 'target3': 'status_t3', 'target4': 'status_t4'})
    status_agg.to_pickle('mlb-merged-data/status.pkl', protocol=4)
    merged = merged.merge(status_agg, on=['statusCode'], how='left')
    print(merged.head())
    print(merged.tail())


def compute_pre_transactions():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    # next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
    #                       (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
    #                       (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27') |
    #                       (next_days['date'] >= '2021-04-01') & (next_days['date'] <= '2021-04-30')]
    transactions = pd.read_pickle('mlb-processed-data/transactions.pkl')
    print(transactions.tail())
    # transactions = transactions[(transactions['date'] >= '2018-03-29') & (transactions['date'] <= '2018-10-01') |
    #                             (transactions['date'] >= '2019-03-20') & (transactions['date'] <= '2019-09-29') |
    #                             (transactions['date'] >= '2020-07-23') & (transactions['date'] <= '2020-09-27') |
    #                             (transactions['date'] >= '2021-04-01') & (transactions['date'] <= '2021-04-30')]
    merged = next_days.merge(transactions, on=['date', 'playerId'], how='left').reset_index(drop=True)
    agg = merged[['typeCode', 'target1', 'target2', 'target3', 'target4']].groupby(['typeCode']).median()
    agg = agg.rename(columns={'target1': 'tran_t1', 'target2': 'tran_t2', 'target3': 'tran_t3', 'target4': 'tran_t4'})
    agg.to_pickle('mlb-merged-data/pre_transactions.pkl', protocol=4)
    print(agg.head())
    print(agg.tail())
    print(agg.shape)
    print(transactions['typeCode'].value_counts())
    # merged = merged.merge(status_agg, on=['statusCode'], how='left')


def compute_pre_award():
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    awards = pd.read_pickle('mlb-processed-data/awards.pkl')
    print(awards.tail())
    merged = next_days.merge(awards, on=['date', 'playerId'], how='left').reset_index(drop=True)
    agg = merged[['awardId', 'target1', 'target2', 'target3', 'target4']].groupby(['awardId']).median()
    agg = agg.rename(columns={'target1': 'award_t1', 'target2': 'award_t2', 'target3': 'award_t3', 'target4': 'award_t4'})
    agg.to_pickle('mlb-merged-data/pre_awards.pkl', protocol=4)
    print(agg, agg.shape)


def compute_pre_position(training=True):
    next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')

    next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
                          (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
                          (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27')]
    #                       #(next_days['date'] >= '2021-04-01') & (next_days['date'] <= '2021-04-30')]
    next_days = next_days[['date', 'playerId', 'target1', 'target2', 'target3', 'target4']]
    player_box = pd.read_pickle('mlb-processed-data/playerBoxScores.pkl')
    if training:
        next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
                              (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
                              (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27')]
        player_box = player_box[(player_box['date'] >= '2018-03-29') & (player_box['date'] <= '2018-10-01') |
                                (player_box['date'] >= '2019-03-20') & (player_box['date'] <= '2019-09-29') |
                                (player_box['date'] >= '2020-07-23') & (player_box['date'] <= '2020-09-27')]
    else:
        next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
                              (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
                              (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27') |
                              (next_days['date'] >= '2021-04-01') & (next_days['date'] <= '2021-04-30')]
        player_box = player_box[(player_box['date'] >= '2018-03-29') & (player_box['date'] <= '2018-10-01') |
                                (player_box['date'] >= '2019-03-20') & (player_box['date'] <= '2019-09-29') |
                                (player_box['date'] >= '2020-07-23') & (player_box['date'] <= '2020-09-27') |
                                (player_box['date'] >= '2021-04-01') & (player_box['date'] <= '2021-04-30')]
    print(player_box.shape)
    print(player_box['teamId'].nunique())
    print(player_box[['positionName', 'positionType', 'positionCode']].value_counts())

    player_box = player_box[['date', 'playerId', 'positionCode']]
    # player_box.groupby(['date', 'playerId']).min()  # A player may have played > 1 game on a day, so take min teamId
    # merged = next_days.merge(player_box, on=['date', 'playerId'], how='left').reset_index(drop=True)
    # merged = merged[['teamId', 'target1', 'target2', 'target3', 'target4']]
    # merged = merged.fillna(0)  # this will fill player_box as 0 for when there was no game played
    # merged_y_game = merged[(merged['teamId'] != np.nan)]
    # merged_n_game = merged[(merged['teamId'] == np.nan)]
    # teams_y_game = merged_y_game.groupby(['teamId']).median()
    merged = next_days.merge(player_box, on=['date', 'playerId'], how='left').reset_index(drop=True)
    merged = merged[['positionCode', 'target1', 'target2', 'target3', 'target4']]
    positions = merged.groupby(['positionCode']).median()
    positions = positions.rename(columns={'target1': 'pos_t1', 'target2': 'pos_t2', 'target3': 'pos_t3', 'target4': 'pos_t4'})
    print(positions)
    positions.to_pickle('mlb-merged-data/positions.pkl', protocol=4)


def compute_last_real_offset():
    index = merged[['date', 'playerId']].sort_values(['playerId', 'date']).reset_index(drop=True)
    # targets = merged[['date', 'playerId', 'target1', 'target2', 'target3', 'target4']]
    targets = merged.drop(
        ['engagementMetricsDate', 'target1', 'target2', 'target3', 'target4', 'target1_med', 'target2_med',
         'target3_med', 'target4_med', 'target1_mean', 'target2_mean', 'target3_mean', 'target4_mean'], 1)
    targets = targets.sort_values(['playerId', 'date']).reset_index(drop=True)
    targets = targets.drop('date', 1)
    targets_1d = targets.groupby(['playerId']).shift(1).fillna(0)
    columns = list(map(lambda x: '1d_' + x, list(targets_1d.columns)))
    index[columns] = targets_1d
    # targets_2d = targets.groupby(['playerId']).shift(2).fillna(0)
    # index[['2d_t1', '2d_t2', '2d_t3', '2d_t4']] = targets_2d
    merged = merged.merge(index, on=['date', 'playerId'], how='left')


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
    # compute_pre_team()
    # compute_pre_features(True)
    # compute_pre_date_features(False)
    # Compute the merged features
    # merged = compute_all_merge_features()
    # saved_merged(merged)
    # compute_pre_award()
    # compute_pre_transactions()
    compute_pre_rolling(False)

    # awards = pd.read_pickle('mlb-processed-data/awards.pkl')
    # twitter = pd.read_pickle('mlb-processed-data/playerTwitterFollowers.pkl')
    # events = pd.read_pickle('mlb-processed-data/events.pkl')
    # standings = pd.read_pickle('mlb-processed-data/standings.pkl')
    # print(awards.tail(), awards.shape)
    # print(twitter.tail(), twitter.shape)
    # print(events.tail(), events.shape)
    # print(standings.tail(), standings.shape)
    # print(events.columns)
    # print(standings.columns)
    # awards['date'].plot()
    # plt.show()

    # player = pd.read_pickle('mlb-processed-data/playerBoxScores.pkl')
    # team = pd.read_pickle('mlb-processed-data/teamBoxScores.pkl')
    # tbf = team_box_features(player, team)
    # tt = tbf[['team_groundIntoTriplePlay', 'team_airOutsPitching', 'team_runsPitching']]
    # print(tt, tt.shape, team.shape)
    # team_box_features(player, team)
    # player_features(players)
    # Test the test flow
    # next_days = pd.read_pickle('mlb-processed-data/nextDayPlayerEngagement.pkl')
    # next_days = next_days[(next_days['date'] >= '2018-03-29') & (next_days['date'] <= '2018-10-01') |
    #                       (next_days['date'] >= '2019-03-20') & (next_days['date'] <= '2019-09-29') |
    #                       (next_days['date'] >= '2020-07-23') & (next_days['date'] <= '2020-09-27') |
    #                       (next_days['date'] >= '2021-04-01') & (next_days['date'] <= '2021-04-30')]
    # merge_offsets(next_days)
    # player_box = pd.read_pickle('mlb-processed-data/playerBoxScores.pkl')
    # pre_agg = pd.read_pickle('mlb-merged-data/pre_train.pkl')
    # compute_merge_features(next_days, player_box, pre_agg, True)
    # Exploring the dates
    # n = next_days[['date', 'target1', 'target2', 'target3', 'target4']].copy()
    # n['target1'] = n['target1'].apply(lambda x: x * 100)
    # n['target2'] = n['target2'].apply(lambda x: x - 0.3)
    # n['target4'] = n['target4'].apply(lambda x: x + 1)
    # n = n[(n['date'] >= '2019-06-20') & (n['date'] <= '2019-07-11')]
    # n = n.groupby(['date']).median()
    # n.plot()
    # plt.show()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
