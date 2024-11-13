import json
import pandas as pd
import numpy as np
import os
import ast


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

        # Create a contingency table for the three variables
        contingency_table = pd.crosstab([df_filtered['stress'], df_filtered['joblose']], df_filtered['satfin'])
        print("Contingency Table:")
        print(contingency_table)

        # Save the contingency table to a CSV file for reference
        contingency_table.to_csv('GSS_contingency_table.csv')
        print("Contingency table saved to 'GSS_contingency_table.csv'")
    except KeyError as e:
        print(f"Error: Some variables are not found in the DataFrame: {e}")
    except Exception as e:
        print(f"Error analyzing selected variables: {e}")
