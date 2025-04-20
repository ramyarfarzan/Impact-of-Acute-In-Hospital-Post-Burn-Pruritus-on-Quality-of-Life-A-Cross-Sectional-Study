import pandas as pd

def summarize_data(df):
    # Summary statistics for numerical variables
    numerical_summary = df.describe().T  # Includes mean, std, min, 25%, 50% (median), 75%, max
    numerical_summary['median'] = df.median()
    
    # Summary statistics for categorical variables
    categorical_summary = {col: df[col].value_counts() for col in df.select_dtypes(include=['object', 'category']).columns}
    
    return numerical_summary, categorical_summary


df = pd.read_csv("Final_data.csv")
numeric_stats, categorical_stats = summarize_data(df)

print("Numerical Variables Summary:")
print(numeric_stats)
print("\nCategorical Variables Summary:")
for col, counts in categorical_stats.items():
    print(f"{col}:")
    print(counts)