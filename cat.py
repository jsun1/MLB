# import pickle
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import datetime
import lightgbm as lgbm
import numpy as np
import pandas as pd
import time

def fit_lgbm(x_train, y_train, x_valid, y_valid, target, training, verbose=100):
    # oof_pred_lgb = np.zeros(len(y_valid), dtype=np.float32)
    # oof_pred_cat = np.zeros(len(y_valid), dtype=np.float32)

    # if os.path.isfile(f'../input/mlb-lgbm-and-catboost-models/model_lgb_{target}.pkl'):
    #     with open(f'../input/mlb-lgbm-and-catboost-models/model_lgb_{target}.pkl', 'rb') as fin:
    #         model = pickle.load(fin)
    # else:

    # model = lgbm.LGBMRegressor(**params)
    # model.fit(x_train, y_train,
    #     eval_set=[(x_valid, y_valid)],
    #     early_stopping_rounds=verbose,
    #     verbose=verbose)
    #
    #     # with open(f'model_lgb_{target}.pkl', 'wb') as handle:
    #     #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # oof_pred_lgb = model.predict(x_valid)
    # score_lgb = mean_absolute_error(oof_pred_lgb, y_valid)
    # print('mae:', score_lgb)

    # if os.path.isfile(f'../input/mlb-lgbm-and-catboost-models/model_cb_{target}.pkl'):
    #     with open(f'../input/mlb-lgbm-and-catboost-models/model_cb_{target}.pkl', 'rb') as fin:
    #         model_cb = pickle.load(fin)
    # else:

    if target == 1:
        estimators = 400
    elif target == 2:
        estimators = 1100
    elif target == 3:
        estimators = 300
    else:
        estimators = 1600
    # estimators = 2000
    learning_rate = 0.05
    model_cb = CatBoostRegressor(
                n_estimators=estimators,  #2000
                learning_rate=learning_rate,
                loss_function='MAE',
                eval_metric='MAE',
                # max_bin=50,
                # subsample=0.9,
                # colsample_bylevel=0.5,
                verbose=100)

    model_cb.fit(x_train, y_train, use_best_model=True,
                     eval_set=(x_valid, y_valid),
                     early_stopping_rounds=50)

    # with open(f'model_cb_{target}.pkl', 'wb') as handle:
    #     pickle.dump(model_cb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    oof_pred_cat = model_cb.predict(x_valid)
    score_cat = mean_absolute_error(oof_pred_cat, y_valid)
    print('mae:', score_cat)

    # if not training:
    #     date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #     link = 'saved-models/' + date + '_c' + str(target) + '.txt'
    #     model_cb.save_model(link)
    #     print('saved at ' + link)

    # return oof_pred_lgb, model, oof_pred_cat, model_cb, score_lgb, score_cat
    return oof_pred_cat, model_cb, score_cat


def model_lgbm_inputs(merged):
    inputs = merged.drop(['date', 'engagementMetricsDate', 'playerId', 'target1', 'target2', 'target3', 'target4'], 1)
    return inputs


def main():
    merged = pd.read_pickle('mlb-merged-data/merged.pkl')
    split_date = pd.to_datetime('2021-04-01')
    training = False
    if training:
        merged_train = merged.loc[merged.date < split_date]
    else:
        merged_train = merged  # train on all data
    merged_val = merged.loc[merged.date >= split_date]
    x_train = model_lgbm_inputs(merged_train)
    x_val = model_lgbm_inputs(merged_val)
    y_train = merged_train[['target1', 'target2', 'target3', 'target4']]
    y_val = merged_val[['target1', 'target2', 'target3', 'target4']]

    # # training lightgbm
    # params = {
    # 'boosting_type': 'gbdt',
    # 'objective':'mae',
    # 'subsample': 0.5,
    # 'subsample_freq': 1,
    # 'learning_rate': 0.03,
    # 'num_leaves': 2**11-1,
    # 'min_data_in_leaf': 2**12-1,
    # 'feature_fraction': 0.5,
    # 'max_bin': 100,
    # 'n_estimators': 2500,
    # 'boost_from_average': False,
    # "random_seed":42,
    # }

    oof_pred_cat1, model_cb1, score_cat1 = fit_lgbm(
        x_train, y_train['target1'],
        x_val, y_val['target1'],
        1, training
    )

    oof_pred_cat2, model_cb2, score_cat2 = fit_lgbm(
        x_train, y_train['target2'],
        x_val, y_val['target2'],
        2, training
    )

    oof_pred_cat3, model_cb3, score_cat3 = fit_lgbm(
        x_train, y_train['target3'],
        x_val, y_val['target3'],
        3, training
    )

    oof_pred_cat4, model_cb4, score_cat4 = fit_lgbm(
        x_train, y_train['target4'],
        x_val, y_val['target4'],
        4, training
    )

    # score = (score_lgb1+score_lgb2+score_lgb3+score_lgb4) / 4
    # print('LightGBM score: {score}')

    score = (score_cat1+score_cat2+score_cat3+score_cat4) / 4
    print('Catboost score:', score)

    # save the models
    if not training:
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        for i, model in enumerate([model_cb1, model_cb2, model_cb3, model_cb4]):
            link = 'saved-models/' + date + '_c' + str(i + 1) + '.txt'
            model.save_model(link)
            print('Saved at', link)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
