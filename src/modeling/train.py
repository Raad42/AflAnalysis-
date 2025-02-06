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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import os
import joblib

file_path = os.path.join('data', 'processed', '12_23data.csv')
mainDF = pd.read_csv(file_path)

#get rid of finals games  
mainDF = mainDF[mainDF.isFinal != 1]
mainDF = mainDF.dropna() 


X = mainDF[['Year', 'Round', 'MaxTemp', 'StadiumCapacity','Rivalry','Rainfall', 'VenueEncode', 'HomeTeamEncode', 'AwayTeamEncode', 'Day', 'MinutesSinceMidnight', 'HomeProbability', 'previous_game_home_position','previous_game_away_position', 'previous_game_home_win_loss', 'previous_game_away_win_loss']]
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

#Deep Neural Network
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader for batch training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

train_data, val_data, train_labels, val_labels = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.1, random_state=42
)

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -------------------------------
# Define the objective function for Optuna
def objective(trial):
    # Sample hyperparameters
    n_units1 = trial.suggest_int('n_units1', 64, 256)
    n_units2 = trial.suggest_int('n_units2', 32, 128)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    
    # Define a simple DNN model using these hyperparameters
    class CustomDNN(nn.Module):
        def __init__(self, input_size):
            super(CustomDNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, n_units1),
                nn.BatchNorm1d(n_units1),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(n_units1, n_units2),
                nn.BatchNorm1d(n_units2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(n_units2, 1)
            )
        def forward(self, x):
            return self.model(x)
    
    input_size = X_train_tensor.shape[1]
    model = CustomDNN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model for a small number of epochs (for speed during search)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate on the validation set
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)
    avg_val_loss = total_loss / total_samples
    return avg_val_loss

# Run the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_trial.params)

# -------------------------------
# Train the final DNN model using the best hyperparameters on the entire training data (train + validation)
best_params = study.best_trial.params
n_units1 = best_params['n_units1']
n_units2 = best_params['n_units2']
dropout_rate = best_params['dropout_rate']
lr = best_params['lr']

class AttendanceDNN(nn.Module):
    def __init__(self, input_size):
        super(AttendanceDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, n_units1),
            nn.BatchNorm1d(n_units1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_units1, n_units2),
            nn.BatchNorm1d(n_units2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_units2, 1)
        )
    def forward(self, x):
        return self.model(x)

input_size = X_train_tensor.shape[1]
final_model = AttendanceDNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=lr)

# Combine the training and validation sets for final training
full_dataset = TensorDataset(torch.cat([train_data, val_data]), torch.cat([train_labels, val_labels]))
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

final_epochs = 100
final_losses = []

for epoch in range(final_epochs):
    final_model.train()
    running_loss = 0.0
    for batch_X, batch_y in full_loader:
        optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    epoch_loss = running_loss / len(full_loader.dataset)
    final_losses.append(epoch_loss)
    print(f"Final Training Epoch {epoch+1}/{final_epochs}, Loss: {epoch_loss:.4f}")

# Plot the final training loss curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(final_losses, label='Final Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final Training Loss Curve')
plt.legend()
plt.show()

# Save the optimized DNN model
DNN_Model_path = os.path.join('models', 'DNN.pkl')
joblib.dump(final_model, DNN_Model_path)

#Categorical Boosting
categorical_features = ['Venue', 'HomeTeam', 'AwayTeam', 'DayC']
numerical_features = ['Year', 'Round', 'MaxTemp', 'Rainfall', 'StadiumCapacity', 'Rivalry', 'MinutesSinceMidnight', 'HomeProbability', 'previous_game_home_position', 'previous_game_away_position', 'previous_game_home_win_loss', 'previous_game_away_win_loss']

for col in categorical_features:
    mainDF[col] = mainDF[col].astype('category')

X = mainDF[['Year', 'Round', 'MaxTemp', 'Rainfall', 'StadiumCapacity', 'Rivalry', 'Venue', 'HomeTeam', 'AwayTeam', 'DayC', 'MinutesSinceMidnight', 'HomeProbability', 'previous_game_home_position','previous_game_away_position', 'previous_game_home_win_loss', 'previous_game_away_win_loss']]
Y = mainDF['Attendance']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.10, random_state=42)

scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

X_test.to_csv('X_testCat.csv', index=False)
y_test.to_csv('y_testCat.csv', index=False) 


cbr = CatBoostRegressor(loss_function='RMSE', random_state=42, cat_features=categorical_features)

scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Define the parameter distributions for Random Search
param_dist = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 6]
}

# Perform Randomized Search with cross-validation
CAT_optim = RandomizedSearchCV(cbr, param_distributions=param_dist, n_iter=10, cv=5, scoring=scorer, n_jobs=-1, random_state=42)
CAT_optim.fit(X_train, y_train, cat_features=categorical_features)

Cat_Model_Path = os.path.join('models', 'CatBoosting.pkl')
joblib.dump(CAT_optim, Cat_Model_Path)

import lightgbm as lgb
from lightgbm import LGBMRegressor

lgb_model = lgb.LGBMRegressor(objective='regression', metric='rmse', boosting_type='gbdt', cat_features=categorical_features)

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