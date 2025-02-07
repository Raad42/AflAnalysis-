import pandas as pd
import numpy as np  
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


class AttendanceDNN(nn.Module):
    def __init__(self, input_size):
        super(AttendanceDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  # First hidden layer with 128 neurons
            nn.ReLU(),
            nn.Linear(128, 64),          # Second hidden layer with 64 neurons
            nn.ReLU(),
            nn.Linear(64, 32),           # Third hidden layer with 32 neurons
            nn.ReLU(),
            nn.Linear(32, 16),           # Fourth hidden layer with 16 neurons
            nn.ReLU(),
            nn.Linear(16, 1)             # Output layer
        )

    def forward(self, x):
        return self.model(x)

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

X_testCat = pd.read_csv('X_testCat.csv')
y_testCat = pd.read_csv('Y_testCat.csv').squeeze()

y_test = pd.to_numeric(y_test, errors='coerce')
y_testCat = pd.to_numeric(y_testCat, errors='coerce')

# Convert DataFrames to tensors
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

categorical_features = ['Venue', 'HomeTeam', 'AwayTeam', 'DayC']
X_testCat[categorical_features] = X_testCat[categorical_features].astype('category')
cat_feature_indices = [X_testCat.columns.get_loc(col) for col in categorical_features]

# Load models
GBR_path = os.path.join('models', 'GradientBoosting.pkl')
CAT_path = os.path.join('models', 'CatBoosting.pkl')
LGBM_path = os.path.join('models', 'LightGBM.pkl')
DNN_path = os.path.join('models', 'DNN.pkl')

GBR_model = joblib.load(GBR_path)
CAT_model = joblib.load(CAT_path)
LGBM_model = joblib.load(LGBM_path)
DNN_model = joblib.load(DNN_path)

# Predict with each non-DNN model
y_predGBR = GBR_model.predict(X_test.to_numpy())
y_predCAT = CAT_model.predict(X_testCat)
y_predLGBM = LGBM_model.predict(X_testCat)

# DNN Model Prediction
DNN_model.eval()
with torch.no_grad():
    
    test_predictions = DNN_model(X_test_tensor)
    
    
    test_predictions = test_predictions.numpy().flatten()
    y_test_actual = y_test_tensor.numpy().flatten()

    # Calculate evaluation metrics for DNN: MSE, MAE, RMSE, and MAPE
    mse_dnn = mean_squared_error(y_test_actual, test_predictions)
    mae_dnn = mean_absolute_error(y_test_actual, test_predictions)
    rmse_dnn = np.sqrt(mse_dnn)
    mape_dnn = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
    print(f"DNN Test MSE: {mse_dnn:.4f}")
    print(f"DNN Test MAE: {mae_dnn:.4f}")
    print(f"DNN Test RMSE: {rmse_dnn:.4f}")
    print(f"DNN Test MAPE: {mape_dnn:.2f}%")

    results_df = pd.DataFrame({
        'Actual Attendance': y_test_actual,
        'Predicted Attendance': test_predictions
    })

    print("\nActual vs Predicted Attendance (DNN):")
    print(results_df.head())


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mse, rmse, r2, mape


metrics_GBR = calculate_metrics(y_test, y_predGBR)
metrics_CAT = calculate_metrics(y_testCat, y_predCAT)
metrics_LGBM = calculate_metrics(y_testCat, y_predLGBM)
metrics_DNN = calculate_metrics(y_test_actual, test_predictions)


metrics_df = pd.DataFrame({
    'Model': ['Gradient Boosting', 'CatBoost', 'LightGBM', 'DNN'],
    'MAE': [metrics_GBR[0], metrics_CAT[0], metrics_LGBM[0], metrics_DNN[0]],
    'MSE': [metrics_GBR[1], metrics_CAT[1], metrics_LGBM[1], metrics_DNN[1]],
    'RMSE': [metrics_GBR[2], metrics_CAT[2], metrics_LGBM[2], metrics_DNN[2]],
    'R2': [metrics_GBR[3], metrics_CAT[3], metrics_LGBM[3], metrics_DNN[3]],
    'MAPE (%)': [metrics_GBR[4], metrics_CAT[4], metrics_LGBM[4], metrics_DNN[4]]
})

# Save the metrics to a CSV file
metrics_df.to_csv('model_evaluation_metrics.csv', index=False)
print("\nEvaluation metrics saved to model_evaluation_metrics.csv")

# Save predictions vs actual values plots for each model
def save_prediction_plot(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.6, color="blue", label="Predictions")
    
    y_min, y_max = min(y_true), max(y_true)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    plt.plot([y_min, y_max], [y_min, y_max], color='red', linestyle='--', label="Ideal")
    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{model_name}_predictions_vs_actual.png")
    plt.close()

# Save plots for each model
save_prediction_plot(y_test, y_predGBR, 'Gradient_Boosting')
save_prediction_plot(y_testCat, y_predCAT, 'CatBoost')
save_prediction_plot(y_testCat, y_predLGBM, 'LightGBM')

def save_venue_colored_plot(y_true, y_pred, venues, model_name):
    plt.figure(figsize=(8, 6))
    plot_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Venue': venues
    })
    
    sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue='Venue',
                    palette='viridis', alpha=0.7)
    
    y_min, y_max = plot_df['Actual'].min(), plot_df['Actual'].max()
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    plt.plot([y_min, y_max], [y_min, y_max], color='red', linestyle='--', label='Ideal')
    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Predictions vs Actual by Venue")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{model_name}_predictions_vs_actual_by_venue.png")
    plt.close()

venues = X_testCat['Venue']
save_venue_colored_plot(y_testCat, y_predCAT, venues, 'CatBoost')
save_venue_colored_plot(y_testCat, y_predLGBM, venues, 'LightGBM')

print("Graphs have been saved as PNG files.")

def group_metrics(df, group_col):
    rows = []
    for name, group in df.groupby(group_col):
        r2 = r2_score(group['Actual'], group['Predicted']) if len(group) > 1 else None
        mae = mean_absolute_error(group['Actual'], group['Predicted'])
        mse = mean_squared_error(group['Actual'], group['Predicted'])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((group['Actual'] - group['Predicted']) / group['Actual'])) * 100
        rows.append({
            group_col: name,
            'Count': len(group),
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE (%)': mape
        })
    return pd.DataFrame(rows)


venue_metrics = group_metrics(eval_df := (X_testCat.assign(Actual=y_testCat, Predicted=y_predCAT)), 'Venue')
home_team_metrics = group_metrics(eval_df, 'HomeTeam')
away_team_metrics = group_metrics(eval_df, 'AwayTeam')

# -------------------------------
# Save the evaluation metrics to CSV files
# -------------------------------
venue_metrics.to_csv('evaluation_metrics_by_venue.csv', index=False)
home_team_metrics.to_csv('evaluation_metrics_by_home_team.csv', index=False)
away_team_metrics.to_csv('evaluation_metrics_by_away_team.csv', index=False)
