# MLB Player Digital Engagement Forecasing

In this [Kaggle competition](https://www.kaggle.com/competitions/mlb-player-digital-engagement-forecasting/overview/description), we’ll predict how fans engage with MLB players’ digital content on a daily basis for a future date range. We have access to player performance data, social media data, and team factors like market size. Successful models will provide new insights into what signals most strongly correlate with and influence engagement.

## Approach

The final model was an ensemble model of three separately-trained models: a basic neural network, a light gradient boosted machine, and a CAT boosted machine. 
Because the loss function of the competition was to predict the mean error (not mean-squared error), the ensemble model used the median (as opposed to mean) of the model outputs. 
The neural network contained one hidden layer and a relu activation function, since the target values were bounded in the 0-100 range. Most target values were near 0. 
The light gradient boosted machine (LGBM) model perfomed the best on the task, although since it outputs only one value, four LGBMs were trained, one for each target value. 
The CATBoost is a variant of the LGBM that performs better on categorical values. Additionally, one other important competition detail was to include lag-based features. 
The lag-based features would input the target value from previous days into the feature set of subsequent days. 

## Results

Ranked #32 (top 4%) as a solo team in the [leaderboard](https://www.kaggle.com/competitions/mlb-player-digital-engagement-forecasting/leaderboard), with a final score of 1.3359.

<img width="1189" alt="Screen Shot 2022-08-13 at 6 16 11 PM" src="https://user-images.githubusercontent.com/3321825/184518516-442ed16d-6351-43e2-bc1b-326920513637.png">
