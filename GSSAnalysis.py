import json
import pandas as pd
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
        print("Generating descriptive statistics for selected variables...")
        # List of variables to examine
        variables = [
            'year', 'hrs1', 'hrs2', 'wrkslf', 'occ10', 'happy', 'joblose', 'satjob', 'class_', 
            'satfin', 'tvhours', 'stress', 'realinc', 'mntlhlth'
        ]
        # Select only the relevant columns and drop missing values
        df_subset = df[variables].copy()
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
