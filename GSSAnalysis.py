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
        df_subset = df[variables]

        # Check the unique values present in each of the selected columns and count them
        for variable in variables:
            unique_values = df_subset[variable].value_counts(dropna=False)
            print(f"Unique values for '{variable}' and their counts:")
            print(unique_values)
    except KeyError as e:
        print(f"Error: Some variables are not found in the DataFrame: {e}")
    except Exception as e:
        print(f"Error analyzing selected variables: {e}")