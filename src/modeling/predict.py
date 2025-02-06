import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

X_testCat = pd.read_csv('X_testCat.csv')
y_testCat = pd.read_csv('Y_testCat.csv').squeeze()

y_test = pd.to_numeric(y_test, errors='coerce')
y_testCat = pd.to_numeric(y_testCat, errors='coerce')

categorical_features = ['Venue', 'HomeTeam', 'AwayTeam', 'DayC']
X_testCat[categorical_features] = X_testCat[categorical_features].astype('category')
cat_feature_indices = [X_testCat.columns.get_loc(col) for col in categorical_features]
# Load models
GBR_path = os.path.join('models', 'GradientBoosting.pkl')
CAT_path = os.path.join('models', 'CatBoosting.pkl')
LGBM_path = os.path.join('models', 'LightGBM.pkl')

GBR_model = joblib.load(GBR_path)
CAT_model = joblib.load(CAT_path)
LGBM_model = joblib.load(LGBM_path)

# Predict with each model
y_predGBR = GBR_model.predict(X_test.to_numpy())
y_predCAT = CAT_model.predict(X_testCat)
y_predLGBM = LGBM_model.predict(X_testCat)

# Define evaluation metrics function
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Calculate metrics for each model
metrics_GBR = calculate_metrics(y_test, y_predGBR)
metrics_CAT = calculate_metrics(y_testCat, y_predCAT)
metrics_LGBM = calculate_metrics(y_testCat, y_predLGBM)

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
def save_prediction_plot(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    
    # Plot predictions vs. actual values
    plt.scatter(y_true, y_pred, alpha=0.6, color="blue", label="Predictions")
    
    # Calculate min and max for the diagonal line
    y_min, y_max = min(y_true), max(y_true)
    if y_min == y_max:
        # Adjust if there's no range in actual values
        y_min -= 1
        y_max += 1

    # Plot the perfect prediction line
    plt.plot([y_min, y_max], [y_min, y_max], color='red', linestyle='--', label="Ideal")
    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f"{model_name}_predictions_vs_actual.png")
    plt.close()  # Close the figure

# Save plots for each model
save_prediction_plot(y_test, y_predGBR, 'Gradient_Boosting')
save_prediction_plot(y_testCat, y_predCAT, 'CatBoost')
save_prediction_plot(y_testCat, y_predLGBM, 'LightGBM')

def save_venue_colored_plot(y_true, y_pred, venues, model_name):
    plt.figure(figsize=(8, 6))
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Venue': venues
    })
    
    # Use seaborn to scatter plot with venue as hue
    sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue='Venue',
                    palette='viridis', alpha=0.7)
    
    # Draw the ideal (diagonal) line
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

eval_df = X_testCat.copy()
eval_df['Actual'] = y_testCat.values
eval_df['Predicted'] = y_predCAT

# -------------------------------
# Define a helper function to compute metrics for a given grouping column
# -------------------------------
def group_metrics(df, group_col):
    rows = []
    # Group by the specified column (e.g., 'Venue', 'HomeTeam', or 'AwayTeam')
    for name, group in df.groupby(group_col):
        # Only compute R2 if there is more than one sample; otherwise, set it to None.
        r2 = r2_score(group['Actual'], group['Predicted']) if len(group) > 1 else None
        rows.append({
            group_col: name,
            'Count': len(group),
            'MAE': mean_absolute_error(group['Actual'], group['Predicted']),
            'MSE': mean_squared_error(group['Actual'], group['Predicted']),
            'R2': r2
        })
    return pd.DataFrame(rows)

# -------------------------------
# Compute metrics for each venue, home team, and away team
# -------------------------------
venue_metrics = group_metrics(eval_df, 'Venue')
home_team_metrics = group_metrics(eval_df, 'HomeTeam')
away_team_metrics = group_metrics(eval_df, 'AwayTeam')

# -------------------------------
# Save the evaluation metrics to CSV files
# -------------------------------
venue_metrics.to_csv('evaluation_metrics_by_venue.csv', index=False)
home_team_metrics.to_csv('evaluation_metrics_by_home_team.csv', index=False)
away_team_metrics.to_csv('evaluation_metrics_by_away_team.csv', index=False)
