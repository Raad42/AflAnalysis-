import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Load models
GBR_path = os.path.join('models', 'GradientBoosting.pkl')
CAT_path = os.path.join('models', 'CatBoosting.pkl')
LGBM_path = os.path.join('models', 'LightGBM.pkl')

GBR_model = joblib.load(GBR_path)
CAT_model = joblib.load(CAT_path)
LGBM_model = joblib.load(LGBM_path)

# Predict with each model
y_predGBR = GBR_model.predict(X_test)
y_predCAT = CAT_model.predict(X_test)
y_predLGBM = LGBM_model.predict(X_test)

# Define evaluation metrics function
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Calculate metrics for each model
metrics_GBR = calculate_metrics(y_test, y_predGBR)
metrics_CAT = calculate_metrics(y_test, y_predCAT)
metrics_LGBM = calculate_metrics(y_test, y_predLGBM)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame({
    'Model': ['Gradient Boosting', 'CatBoost', 'LightGBM'],
    'MAE': [metrics_GBR[0], metrics_CAT[0], metrics_LGBM[0]],
    'MSE': [metrics_GBR[1], metrics_CAT[1], metrics_LGBM[1]],
    'R2': [metrics_GBR[2], metrics_CAT[2], metrics_LGBM[2]],
})

# Save the metrics to a CSV file
metrics_df.to_csv('model_evaluation_metrics.csv', index=False)

# Save predictions vs actual values plots for each model
def save_prediction_plot(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'{model_name}: Predictions vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions_vs_actual.png')
    plt.close()  # Close the plot to prevent displaying it

# Save plots for each model
save_prediction_plot(y_test, y_predGBR, 'Gradient_Boosting')
save_prediction_plot(y_test, y_predCAT, 'CatBoost')
save_prediction_plot(y_test, y_predLGBM, 'LightGBM')

print("Graphs have been saved as PNG files.")


