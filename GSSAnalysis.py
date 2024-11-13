import json
import pandas as pd
import numpy as np
import os
import ast

from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path to the CSV file
gss_csv_file = 'GSS_cumulative_data.csv'

# Load the CSV file into a pandas DataFrame
try:
    print("Attempting to load the CSV file...")
    df = pd.read_csv(gss_csv_file)
    print("Successfully loaded GSS cumulative data.")
    print("DataFrame preview:")
    print(df.head())
except Exception as e:
    print(f"Error loading CSV file: {e}")
    df = None

# Analyze intersectional variables if DataFrame is available
if df is not None:
    try:
        print("Selecting key variables for analysis...")
        # Pick 3 key variables to examine based on potential insights into freeze and stress
        variables = ['stress', 'joblose', 'satfin']

        # Replace inapplicable and non-answer values with NaN for analysis
        df_subset = df[variables].replace({
            r'\.i:.*': np.nan, 
            r'\.n:.*': np.nan, 
            r'\.d:.*': np.nan, 
            r'\.s:.*': np.nan,
            r'\.y:.*': np.nan
        }, regex=True)

        # Drop rows with any NaN values to only keep rows with complete data
        df_filtered = df_subset.dropna()

        # Display the filtered data
        print("Filtered DataFrame preview:")
        print(df_filtered.head())

        # Perform Spearman correlation analysis on the filtered data
        print("Calculating Spearman correlation between selected variables...")
        spearman_corr, p_value = spearmanr(df_filtered, axis=0)
        spearman_corr_df = pd.DataFrame(spearman_corr, index=variables, columns=variables)
        print("Spearman Correlation Matrix:")
        print(spearman_corr_df)

        # Save the Spearman correlation matrix to a CSV file for reference
        spearman_corr_df.to_csv('GSS_spearman_correlation.csv')
        print("Spearman correlation matrix saved to 'GSS_spearman_correlation.csv'")

        # Plot the Spearman correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(spearman_corr_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
        plt.title('Spearman Correlation Matrix for Selected Variables')
        plt.savefig('GSS_spearman_correlation_heatmap.png')
        plt.show()
        print("Spearman correlation heatmap saved to 'GSS_spearman_correlation_heatmap.png'")
    except KeyError as e:
        print(f"Error: Some variables are not found in the DataFrame: {e}")
    except Exception as e:
        print(f"Error analyzing selected variables: {e}")
