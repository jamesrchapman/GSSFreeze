import json
import pandas as pd
import os
import zipfile
import numpy as np

stata_folder = 'STATA'
unzip_directory = 'Unzipped_GSS'

# Read all .dta files into a pandas DataFrame
all_dataframes = []

for filename in os.listdir(unzip_directory):
    if filename.endswith('.dta'):
        try:
            file_path = os.path.join(unzip_directory, filename)
            print(f"Processing file: {filename}")
            
            # Extract the year from the filename if possible
            year = filename.split('GSS')[-1].split('.dta')[0]
            print(f"Extracted year: {year}")

            # Read the Stata file without converting categoricals to avoid errors
            df = pd.read_stata(file_path, convert_categoricals=False)
            print(f"Loaded DataFrame shape: {df.shape}")

            # Add year column to the DataFrame
            df['year'] = year
            all_dataframes.append(df)
        
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Combine all DataFrames into one if there are any DataFrames to combine
if all_dataframes:
    # Align all DataFrames by reindexing to handle different shapes
    all_columns = sorted(set().union(*(df.columns for df in all_dataframes)))
    aligned_dataframes = [df.reindex(columns=all_columns, fill_value=np.nan) for df in all_dataframes]
    
    combined_df = pd.concat(aligned_dataframes, ignore_index=True)
    # Display the combined DataFrame to verify the data
    print("Combined DataFrame shape:", combined_df.shape)
    print(combined_df.head())
    # Optionally, save the combined DataFrame to a CSV file for easier use later
    combined_df.to_csv('combined_GSS_data.csv', index=False)
else:
    print("No DataFrames were loaded, nothing to combine.")