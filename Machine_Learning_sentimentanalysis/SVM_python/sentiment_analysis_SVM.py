import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("/Users/roujinan/Desktop/Data/merged_data.csv")

# Step 2: Select features and target
features = [
    "Undergraduate.Tuition", "Graduate.Tuition", "Acceptance.Rate", "SAT", "ACT",
    "Avg..F", "Avg..C", "weather_rank_h_l", "crime_2018", "crime_2020", 
    "crime_2021", "crime_2022", "US_News_rank_2025", "Change_in_rank"
]
target = "average_sentiment"

ml_data = data[features + [target]].dropna()

X = ml_data[features]
y = ml_data[target]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Step 4: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train SVM with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}
svm = SVR()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Step 6: Get the best model
best_svm = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Step 7: Predict on the test set
y_pred = best_svm.predict(X_test_scaled)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"SVM Mean Squared Error (MSE): {mse}")
print(f"SVM R² Score: {r2}")

# Optional: Save predictions
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv("svm_predictions.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Extract grid search results
results = pd.DataFrame(grid_search.cv_results_)

# Visualize MSE as a heatmap for `C` and `gamma` when kernel='rbf'
heatmap_data = results[results['param_kernel'] == 'rbf']
pivot_table = heatmap_data.pivot_table(
    values='mean_test_score', 
    index='param_gamma', 
    columns='param_C'
)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("Grid Search MSE for RBF Kernel")
plt.xlabel("C")
plt.ylabel("Gamma")
plt.savefig("/Users/roujinan/Desktop/Data/Grid Search MSE for RBF Kernel.png")
plt.close()



# Compute residuals
residuals = y_test - y_pred

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig("/Users/roujinan/Desktop/Data/Residuals Plot.png")
plt.close()

# Sort residuals
sorted_residuals = np.sort(residuals)
cumulative_probabilities = np.linspace(0, 1, len(sorted_residuals))

# Cumulative distribution plot
plt.figure(figsize=(8, 6))
plt.plot(sorted_residuals, cumulative_probabilities, color='blue')
plt.title('Cumulative Error Distribution')
plt.xlabel('Residuals')
plt.ylabel('Cumulative Probability')
plt.grid()
plt.savefig("/Users/roujinan/Desktop/Data/Cumulative Error Distribution.png")
plt.close()


from sklearn.metrics import roc_curve, auc


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', lw=2)
plt.title('Actual vs Predicted Sentiments')
plt.xlabel('Actual Sentiment')
plt.ylabel('Predicted Sentiment')
plt.grid()
plt.savefig("/Users/roujinan/Desktop/Data/Actual vs Predicted Sentiments.png")
plt.close()


import shap

# Initialize SHAP Explainer
explainer = shap.KernelExplainer(best_svm.predict, X_train_scaled)
shap_values = explainer.shap_values(X_test_scaled, nsamples=100)

# Summary Plot
shap.summary_plot(shap_values, X_test, show=False)  # Do not display in the terminal
plt.title("SHAP Summary Plot for SVM")
plt.savefig("/Users/roujinan/Desktop/Data/shap_summary_plot.png")  # Save the plot
plt.close()  # Close the plot to prevent overlapping issues

# Dependence Plot Example
shap.dependence_plot("SAT", shap_values, X_test, show=False)
plt.title("SHAP Dependence Plot for SAT")
plt.savefig("/Users/roujinan/Desktop/Data/shap_dependence_plot.png")  # Save the plot
plt.close()

# Force Plot (Static Image)
force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
plt.title("SHAP Force Plot")
plt.savefig("/Users/roujinan/Desktop/Data/shap_force_plot.png")  # Save the force plot
plt.close()