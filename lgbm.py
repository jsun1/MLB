import datetime
from sklearn.metrics import mean_absolute_error
import lightgbm as lgbm
import numpy as np
import pandas as pd
import time


def fit_lgbm(x_train, y_train, x_valid, y_valid, params=None, verbose=100):
    oof_pred = np.zeros(len(y_valid), dtype=np.float32)
    model = lgbm.LGBMRegressor(**params)
    model.fit(x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        early_stopping_rounds=verbose,
        verbose=verbose)
    oof_pred = model.predict(x_valid)
    score = mean_absolute_error(oof_pred, y_valid)
    print('mae:', score)
    return oof_pred, model, score


# Put this in the Notebook!
def model_lgbm_inputs(merged):
    # inputs = merged.drop(['date', 'engagementMetricsDate', 'playerId', 'jerseyNum', 'target1', 'target2', 'target3', 'target4', 'year', 'dayOfWeek', 'day', 'week'], 1).to_numpy(dtype=np.float32)
    inputs = merged.drop(['date', 'engagementMetricsDate', 'playerId', 'target1', 'target2', 'target3', 'target4'], 1)#.to_numpy(dtype=np.float32)
    return inputs


def main():
    merged = pd.read_pickle('mlb-merged-data/merged.pkl')
    split_date = pd.to_datetime('2021-06-10')
    training = False
    if training:
        merged_train = merged.loc[merged.date < split_date]
    else:
        merged_train = merged  # train on all data
    merged_val = merged.loc[merged.date >= split_date]
    x_train = model_lgbm_inputs(merged_train)
    x_val = model_lgbm_inputs(merged_val)
    y_train = merged_train[['target1', 'target2', 'target3', 'target4']]#.to_numpy(dtype=np.float32)
    y_val = merged_val[['target1', 'target2', 'target3', 'target4']]#.to_numpy(dtype=np.float32)

    # training lightgbm
    # params = {
    #     'objective': 'mae',
    #     'reg_alpha': 0.1,
    #     'reg_lambda': 0.1,
    #     'n_estimators': 100000,
    #     'learning_rate': 0.1,
    #     'random_state': 42,
    # }

    # num_leaves (default=31)
    # n_estimators (default=100)
    # min_child_samples (default=20)

    params1 = {
        'objective': 'mae',
        # 'reg_alpha': 0.1, #0.14947461820098767,
        # 'reg_lambda': 0.1, #0.10185644384043743,
        'n_estimators': 300, #3633,
        # 'learning_rate': 0.08046301304430488,
        'num_leaves': 300,  #674,
        # 'feature_fraction': 0.9101240539122566,
        # 'bagging_fraction': 0.9884451442950513,
        # 'bagging_freq': 8,
    }

    params2 = {
        'objective': 'mae',
        # 'reg_alpha': 0.1,
        # 'reg_lambda': 0.1,
        'n_estimators': 100,  # 80,
        # 'learning_rate': 0.1,
        # 'random_state': 42,
        'num_leaves': 300,  # 22
    }

    params3 = {
        'objective': 'mae',
        # 'reg_alpha': 0.1,
        # 'reg_lambda': 0.1,
        'n_estimators': 300, #10000,
        # 'learning_rate': 0.1,
        # 'random_state': 42,
        'num_leaves': 300,
    }

    params4 = {
        'objective': 'mae',
        # 'reg_alpha': 0.1,
        # 'reg_lambda': 0.1,
        'n_estimators': 100, #10000,
        # 'learning_rate': 0.1,
        # 'random_state': 42,
        'num_leaves': 300,
    }

    # params4 = {
    #     'objective': 'mae',
    #     'reg_alpha': 0.1, #0.016468100279441976,
    #     'reg_lambda': 0.09128335764019105,
    #     'n_estimators': 9868,
    #     'learning_rate': 0.10528150510326864,
    #     'num_leaves': 157,
    #     'feature_fraction': 0.5419185713426886,
    #     'bagging_fraction': 0.2637405128936662,
    #     'bagging_freq': 19,
    #     'min_child_samples': 71
    # }

    oof1, model1, score1 = fit_lgbm(
        x_train, y_train['target1'],
        x_val, y_val['target1'],
        params1
    )
    oof2, model2, score2 = fit_lgbm(
        x_train, y_train['target2'],
        x_val, y_val['target2'],
        params2
    )
    oof3, model3, score3 = fit_lgbm(
        x_train, y_train['target3'],
        x_val, y_val['target3'],
        params3
    )
    oof4, model4, score4 = fit_lgbm(
        x_train, y_train['target4'],
        x_val, y_val['target4'],
        params4
    )

    score = (score1 + score2 + score3 + score4) / 4
    print('score', score)
    # save the models
    if not training:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        for i, model in enumerate([model1, model2, model3, model4]):
            link = 'saved-models/' + date + '_m' + str(i+1) + '.txt'
            model.booster_.save_model(link)
            print('Saved at', link)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
