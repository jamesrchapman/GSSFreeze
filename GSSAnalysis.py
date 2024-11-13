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

# Generate descriptive statistics for selected variables if DataFrame is available
if df is not None:
    try:
        print("Checking for non-numeric values in the DataFrame and calculating numeric percentages...")
        numeric_percentages = {}
        numeric_dataframes = []

        # Iterate through the DataFrame to check numeric values and calculate percentage
        for column in df.columns:
            total_count = len(df[column])
            numeric_count = df[column].apply(lambda x: pd.to_numeric(x, errors='coerce')).notna().sum()
            numeric_percentage = (numeric_count / total_count) * 100
            numeric_percentages[column] = numeric_percentage

            # Print percentage of numeric values
            print(f"Column '{column}': {numeric_percentage:.2f}% numeric values")

            # Create a version of the DataFrame with only numeric values for each column
            if numeric_percentage > 0:  # Only process columns that have some numeric data
                numeric_data = pd.to_numeric(df[column], errors='coerce')  # Convert non-numeric to NaN
                numeric_dataframes.append(numeric_data)

        # Combine numeric columns into a new DataFrame
        df_numeric = pd.concat(numeric_dataframes, axis=1).dropna()

        # Save the numeric-only DataFrame to a CSV file for later use
        df_numeric.to_csv('GSS_numeric_data.csv', index=False)
        print("Numeric-only DataFrame saved to 'GSS_numeric_data.csv'")

        print("Generating descriptive statistics for selected variables...")
        # List of variables to examine
        variables = [
            'year', 'hrs1', 'hrs2', 'wrkslf', 'occ10', 'happy', 'joblose', 'satjob', 'class_', 
            'satfin', 'tvhours', 'stress', 'realinc', 'mntlhlth'
        ]
        # Select only the relevant columns and drop rows with NaN values
        df_subset = df[variables].dropna()
        descriptive_table = df_subset.describe(include='all')
        
        # Display the descriptive statistics
        print("Descriptive Statistics Table:")
        print(descriptive_table)
        
        # Optionally, save the descriptive statistics to a CSV file for reference
        descriptive_table.to_csv('GSS_descriptive_statistics.csv')
        print("Descriptive statistics saved to 'GSS_descriptive_statistics.csv'")
    except KeyError as e:
        print(f"Error: Some variables are not found in the DataFrame: {e}")
    except Exception as e:
        print(f"Error generating descriptive statistics: {e}")
