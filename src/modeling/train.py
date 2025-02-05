import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from catboost import CatBoostRegressor
import os
import joblib

file_path = os.path.join('data', 'processed', '12_23data.csv')
mainDF = pd.read_csv(file_path)

mainDF = mainDF.dropna() 

X = mainDF[['Year', 'Round', 'MaxTemp', 'Rainfall', 'Venue', 'HomeTeam', 'AwayTeam', 'Day', 'MinutesSinceMidnight', 'HomeProbability', 'previous_game_home_position','previous_game_away_position', 'previous_game_home_win_loss', 'previous_game_away_win_loss']]
Y = mainDF['Attendance']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.10, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_test_df = pd.DataFrame(X_test, columns=X.columns)
y_test_df = pd.DataFrame(y_test, columns=['Attendance'])  

#make x test and y test csv
X_test_df.to_csv('X_test.csv', index = False)
y_test_df.to_csv('y_test.csv', index = False)


# Define the parameter distributions for Random Search
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
model = GradientBoostingRegressor()

# Perform Randomized Search with cross-validation
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Print the best hyperparameters for Random Search
print("Random Search - Best Hyperparameters:", random_search.best_params_)

# Train the model with the best hyperparameters
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

GBR_Model = os.path.join('models', 'GradientBoosting.pkl')
joblib.dump(best_model,GBR_Model)


cbr = CatBoostRegressor(loss_function='RMSE', random_state=42)

scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Define the parameter distributions for Random Search
param_dist = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 6]
}

# Perform Randomized Search with cross-validation
CAT_optim = RandomizedSearchCV(cbr, param_distributions=param_dist, n_iter=10, cv=5, scoring=scorer, n_jobs=-1, random_state=42)
CAT_optim.fit(X_train, y_train)

Cat_Model_Path = os.path.join('models', 'CatBoosting.pkl')
joblib.dump(CAT_optim, Cat_Model_Path)

import lightgbm as lgb
from lightgbm import LGBMRegressor

lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse', boosting_type='gbdt')

# Define the parameter grid to search
param_dist = {
    'num_leaves': np.arange(8, 64),
    'learning_rate': np.logspace(-4, -1, 10),
    'max_depth': np.arange(3, 15),
    'feature_fraction': np.linspace(0.5, 1.0, 6),
    'bagging_fraction': np.linspace(0.5, 1.0, 6),
    'bagging_freq': np.arange(1, 8),
    'lambda_l1': np.logspace(-5, -1, 5),
    'lambda_l2': np.logspace(-5, -1, 5),
}

# Define the RandomizedSearchCV
LGB_optim = RandomizedSearchCV(lgb_model, param_distributions=param_dist, 
                                   n_iter=50,  # Number of parameter combinations to try
                                   scoring='neg_root_mean_squared_error',  # RMSE is the score metric
                                   cv=3,  # Cross-validation folds
                                   verbose=2,  # Prints the progress
                                   n_jobs=-1,  # Uses all cores
                                   random_state=42)


LGB_optim.fit(X_train, y_train)

LGB_Model_Path = os.path.join('models', 'LightGBM.pkl')
joblib.dump(LGB_optim, LGB_Model_Path)