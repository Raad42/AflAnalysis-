import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# Load dataset
mainDF = pd.read_csv(r'C:\Users\raadr\OneDrive\Desktop\AflAnalysis-\data\interim\H&AsesData.csv')
mainDF = mainDF.dropna()

X = mainDF[['Year', 'Round', 'MaxTemp', 'MinTemp', 'Rainfall', 'Venue', 'HomeTeam', 'AwayTeam', 'Day', 
            'homePosition', 'homePoints', 'MinutesSinceMidnight', 'homePercentage', 'awayPosition']]
Y = mainDF['Attendance']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Save test data for later evaluation
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Scale test data too

# Train model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model training complete. Test data saved.")


mainDF = pd.read_csv(r'C:\Users\raadr\OneDrive\Desktop\AflAnalysis-\data\interim\proCatH&AData.csv')
mainDF.columns.tolist()
#drop null values
mainDF = mainDF.dropna()

categorical_features = ['Venue', 'HomeTeam', 'AwayTeam', 'Day', 'StartTime']
cbr = CatBoostRegressor(loss_function='RMSE', cat_features=categorical_features, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Define the scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)
# Define the parameter distributions for Random Search
param_dist = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [3, 6]
}

# Perform Randomized Search with cross-validation
random_search = RandomizedSearchCV(cbr, param_distributions=param_dist, n_iter=10, cv=5, scoring=scorer, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Print the best hyperparameters for Random Search
print("Random Search - Best Hyperparameters:", random_search.best_params_)
# Plot the results of the Randomized Search
results = pd.DataFrame(random_search.cv_results_)

joblib.dump(random_search, 'trained_model2.pkl')
joblib.dump(scaler, 'scaler.pkl')