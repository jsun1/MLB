# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import gc
from tqdm.auto import tqdm

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

### DATA CONFIG
targets = ['target1','target2','target3','target4']
SPLIT = pd.to_datetime('2020-01-01')
features = ['have_game']
# features = ['have_game', 'battingOrder', 'gamesPlayedBatting', 'flyOuts', 'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns', 'strikeOuts',
#                'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch', 'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
#                'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi', 'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
#                'pickoffs']  # the agg_target1234 are added to this. These are features to use in training
identifiers = ['playerId','date']  # the target1234 are added to this. These are columns to use in the dataframe
print(SPLIT)
### MODEL CONFIG

def compute_metric(ground_truth,predicted):
    # inputs are both pandas.DataFrame
    ground_truth_sorted = ground_truth.sort_values(identifiers).reset_index(drop=True) # reset_index replaces row names w/ 0,1,2.. drop=True means the prev row names are not kept
    predicted_sorted = predicted.sort_values(identifiers).reset_index(drop=True)
    metric = (ground_truth_sorted[targets]-predicted_sorted[targets]).abs().mean()
    metric.loc['CV'] = metric.mean()  # loc accesses the DataFrame at a certain row location
    return metric

def pair_correlation(df1,df2):
    correlation_dfs = pd.merge(df1,df2,on=identifiers).corr() # merges based on identifiers, all other columns get _x, _y suffix? corr is pairwise correlation - gives square matrix
    cols1 = [x for x in df1.columns if x in correlation_dfs.columns and x not in df2.columns]
    cols2 = [x for x in df2.columns if x in correlation_dfs.columns and x not in df1.columns]
    return correlation_dfs.loc[cols1,cols2] # only use rows where the name is a column in either df1 or df2

# READ DATA

# player engagements is the target file
engagements = pd.read_csv('../input/mlb-train-processed-data/nextDayPlayerEngagement.csv',index_col=0).rename({'engagementMetricsDate':'date'},axis=1)
engagements = pd.read_csv('../input/mlb-train-processed-data/nextDayPlayerEngagement.csv',index_col=0).rename({'engagementMetricsDate':'date'},axis=1)
engagements['date'] = pd.to_datetime(engagements['date'])-pd.to_timedelta('1 days') # subtract one day from each date (since these are next-day engagement metrics)
print(engagements.shape)
engagements.head()

player_box_scores = pd.read_csv('../input/mlb-train-processed-data/playerBoxScores.csv',index_col=0).rename(columns={'gameDate':'date'})
player_box_scores = pd.read_csv('/mlb-train-processed-data/playerBoxScores.csv',index_col=0).rename(columns={'gameDate':'date'})
player_box_scores['date'] = pd.to_datetime(player_box_scores['date']) #creates datetime objects from the date string
player_box_scores['have_game'] = 1
print(player_box_scores.shape)
player_box_scores = player_box_scores.drop_duplicates(['playerId','date'],keep='first')
print(player_box_scores.shape)
# player_box_scores['battingOrder'] = player_box_scores['battingOrder'].fillna(1000)
# print(player_box_scores['rbi'].isna().sum())
# player_box_scores['rbi'].hist()

df = pd.merge(engagements,player_box_scores,on=['playerId','date'],how='left') # merge 2 DataFrames, based on playerId& date columns. how=left means only use keys from engagements(L)
df['have_game'] = df['have_game'].fillna(0)
# df['battingOrder'] = df['battingOrder'].fillna(1000)
# df['gamesPlayedBatting'] = df['gamesPlayedBatting'].fillna(0)
# df['flyOuts'] = df['flyOuts'].fillna(0)
# df['groundOuts'] = df['groundOuts'].fillna(0)
# df['runsScored'] = df['runsScored'].fillna(0)
# df['doubles'] = df['doubles'].fillna(0)
# df['triples'] = df['triples'].fillna(0)
# df['homeRuns'] = df['homeRuns'].fillna(0)
# df['strikeOuts'] = df['strikeOuts'].fillna(0)
# df['baseOnBalls'] = df['baseOnBalls'].fillna(0)
# df['intentionalWalks'] = df['intentionalWalks'].fillna(0)
# df['hits'] = df['hits'].fillna(0)
# df['hitByPitch'] = df['hitByPitch'].fillna(0)
# df['atBats'] = df['atBats'].fillna(0)
# df['caughtStealing'] = df['caughtStealing'].fillna(0)
# df['stolenBases'] = df['stolenBases'].fillna(0)
# df['groundIntoDoublePlay'] = df['groundIntoDoublePlay'].fillna(0)
# df['groundIntoTriplePlay'] = df['groundIntoTriplePlay'].fillna(0)
# df['plateAppearances'] = df['plateAppearances'].fillna(0)
# df['totalBases'] = df['totalBases'].fillna(0)
# df['rbi'] = df['rbi'].fillna(0)
# df['leftOnBase'] = df['leftOnBase'].fillna(0)
# df['sacBunts'] = df['sacBunts'].fillna(0)
# df['sacFlies'] = df['sacFlies'].fillna(0)
# df['catchersInterference'] = df['catchersInterference'].fillna(0)
# df['pickoffs'] = df['pickoffs'].fillna(0)
del player_box_scores,engagements
gc.collect()
print(df.shape)
df.head()

correlations = df.corr()[['target1','target2','target3','target4']] # corr makes full square matrix of all correlations. Then we just want target columns
correlations['mean_corr'] = correlations.mean(axis=1) # computes the mean as a new column - the mean is per row(index)
correlations.sort_values('mean_corr',ascending=False).head(30) #sorted list of things that correlate most with mean_corr

# TRUNCATED VALIDATION

## Train Targets
train_targets = df.loc[df.date<SPLIT,identifiers+targets].reset_index(drop=True)
val_targets = df.loc[df.date>=SPLIT,identifiers+targets].reset_index(drop=True)  #make 2 sets, train + val based on date SPLIT. has playerId, date, targets
print('train & val targets:', train_targets.shape,val_targets.shape)

## Train Features
train_features = df.loc[df.date<SPLIT,identifiers+features].reset_index(drop=True)
val_features = df.loc[df.date>=SPLIT,identifiers+features].reset_index(drop=True) #make 2 sets, train + val based on date SPLIT. has playerId, date, have_game
print('train & val feature:', train_features.shape,val_features.shape)

## Compute Aggregate Features From Train
aggregate = train_targets[train_features.have_game==0].groupby('playerId')[targets].median().reset_index() #per player, get median of each target
aggregate.columns = ['agg_'+x if 'target' in x else x for x in aggregate.columns] #prefix "agg_target" for each target column
print(aggregate.shape)
aggregate.head()

train_features = pd.merge(train_features,aggregate,on='playerId')
val_features = pd.merge(val_features,aggregate,on='playerId')  # adds 4 more columns (agg_targets) to train/val_features
print('train & val feature:', train_features.shape,val_features.shape)
print(train_features.date.min(),train_features.date.max(),val_features.date.min(),val_features.date.max())
train_features.sample(3)

# MODELING

class Regressor():

    def fit(self, X, y):
        temp = pd.concat([X, y], axis=1)
        temp.columns = ['have_game', '_', 'target']
        neg, pos = temp.groupby('have_game').target.median().values
        self.offset = (pos - neg)

    def predict(self, X):
        offset = X.values[:, 0] * self.offset
        return np.clip(X.values[:, 1] + offset, 0, 100)

# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
# reg = RandomForestRegressor(n_estimators=100,max_depth=6, random_state=0,n_jobs=-1,verbose=1)
predicted = train_targets.sort_values(identifiers).reset_index(drop=True)[identifiers]
gt = train_targets.sort_values(identifiers).reset_index(drop=True)
regs = {}
for target in targets:
    selected_features = features + ['agg_'+target]
#     reg = Ridge(alpha=1.0)
    reg = Regressor()
    X = train_features.sort_values(identifiers).reset_index(drop=True)[selected_features]
    y = train_targets.sort_values(identifiers).reset_index(drop=True)[target]
    print(len(selected_features), X.shape, y.shape)
    print(X.head())
    reg.fit(X, y)
    print('target', target,reg.offset)
    predicted[target] = reg.predict(X)
    regs[target] = reg
# predicted = pd.DataFrame(predicted,columns=targets)
print("Train Metrics: ",compute_metric(gt,predicted).to_dict())
train_features.head()

pd.merge(predicted,train_features,on=identifiers).groupby('have_game')[targets].mean()

pd.merge(train_targets,train_features,on=identifiers).groupby('have_game')[targets].mean()

predicted = val_targets.sort_values(identifiers).reset_index(drop=True)[identifiers]
for target in tqdm(targets):
    selected_features = features + ['agg_'+target]
    X = val_features.sort_values(identifiers).reset_index(drop=True)[selected_features]
    predicted[target] = regs[target].predict(X)
print(val_targets.shape,predicted.shape)
print("Val Metrics: ",compute_metric(val_targets,predicted).to_dict())  # these are the metrics that we should minimize (target1234, CV) for the val set after SPLIT date

pd.merge(predicted,val_features,on=identifiers).groupby('have_game')[targets].mean()

pd.merge(val_targets,val_features,on=identifiers).groupby('have_game')[targets].mean()

pd.concat([train_targets.groupby('playerId').median().mean(),predicted.mean(),val_targets.groupby('playerId').median().mean()],axis=1)

pd.concat([train_targets.mean(),predicted.mean(),val_targets.mean()],axis=1)

# TRAINING FOR TEST

## Train Targets & Features
train_targets = df[identifiers+targets].reset_index(drop=True)
train_features = df[identifiers+features].reset_index(drop=True)

## Compute Aggregate Features From Train
aggregate = train_targets.groupby('playerId')[targets].median().reset_index()
aggregate.columns = ['agg_'+x if 'target' in x else x for x in aggregate.columns]
train_features = pd.merge(train_features,aggregate,on='playerId')

print(train_features.shape,train_targets.shape)
train_features.sample(3)

pair_correlation(train_targets,train_features)

predicted = train_targets.sort_values(identifiers).reset_index(drop=True)[identifiers]
gt = train_targets.sort_values(identifiers).reset_index(drop=True)  # gt = ground truth
regs = {}
for target in tqdm(targets):
    selected_features = features + ['agg_'+target]
    X = train_features.sort_values(identifiers).reset_index(drop=True)[selected_features]
    y = train_targets.sort_values(identifiers).reset_index(drop=True)[target]
    reg.fit(X, y)
    predicted[target] = reg.predict(X)
    regs[target] = reg
print(predicted.shape, gt.shape, len(regs))
print("Train Metrics: ",compute_metric(gt,predicted).to_dict())  # these are the metrics that (target1234, CV) after training on whole set (training for final submission)
gt.head()

