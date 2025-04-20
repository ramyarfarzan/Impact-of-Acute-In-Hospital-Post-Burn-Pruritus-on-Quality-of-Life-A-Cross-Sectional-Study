import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("Final_data.csv")

# Convert categorical Burn_Depth to numerical
df['Burn_Depth_Num'] = df['Burn_Depth'].map({'Partial-thickness': 0, 'Full-thickness': 1})

# Select predictors and target
X = df[['Itch_Severity', 'Itch_Duration', 'TBSA_Burned', 'Burn_Depth_Num']]
y = df['QoL_Score']

# Linear Regression (sklearn)
lr_model = LinearRegression()
lr_model.fit(X, y)
lr_predictions = lr_model.predict(X)

# OLS Regression (statsmodels)
X_with_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_const).fit()

# Define beautiful style
plt.style.use("seaborn-darkgrid")

# Set universal font style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[['QoL_Score', 'Itch_Severity', 'Itch_Duration', 'TBSA_Burned']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
           square=True, linewidths=0.5, fmt=".2f", annot_kws={"size": 14})
plt.title('Correlation Heatmap', pad=15, fontsize=18, fontweight="bold")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# 2. Scatter plots with regression lines
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Relationships with Quality of Life Score', fontsize=20, fontweight="bold")

# Define plot parameters
scatter_kws = {"alpha": 0.6, "s": 50}  # Adjust marker size and transparency
line_kws = {"color": "red", "linewidth": 2}

# Itch Severity
sns.regplot(data=df, x='Itch_Severity', y='QoL_Score', ax=axes[0,0], 
            scatter_kws=scatter_kws, line_kws=line_kws)
axes[0,0].set_title('Itch Severity vs QoL')

# Itch Duration
sns.regplot(data=df, x='Itch_Duration', y='QoL_Score', ax=axes[0,1], 
            scatter_kws=scatter_kws, line_kws=line_kws)
axes[0,1].set_title('Itch Duration vs QoL')

# TBSA Burned
sns.regplot(data=df, x='TBSA_Burned', y='QoL_Score', ax=axes[1,0], 
            scatter_kws=scatter_kws, line_kws=line_kws)
axes[1,0].set_title('TBSA Burned vs QoL')

# Predicted vs Actual
axes[1,1].scatter(y, lr_predictions, color='dodgerblue', alpha=0.6, s=50)
axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
axes[1,1].set_xlabel('Actual QoL Score')
axes[1,1].set_ylabel('Predicted QoL Score')
axes[1,1].set_title('Predicted vs Actual QoL')

plt.tight_layout()

# 3. Coefficient Plot
coefs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
})
plt.figure(figsize=(10, 6))
sns.barplot(data=coefs, x='Feature', y='Coefficient', palette='magma', edgecolor="black")
plt.title('Regression Coefficients', fontsize=18, fontweight="bold")
plt.xticks(rotation=45)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
