import json
import pandas as pd
import os
import zipfile

# Replace 'your_file.json' with the path to your JSON file
file_path = 'GSS.json'
labels = ['year', 'id_', 'hrs1', 'hrs2', 'wrkslf', 'occ10', 'sphrs1', 'sphrs2', 'happy', 'hapmar', 'joblose', 'satjob', 'class_', 'satfin', 'finalter', 'tvhours', 'wrktype', 'yearsjob', 'waypaid', 'wrksched', 'moredays', 'mustwork', 'wrkhome', 'whywkhme', 'famwkoff', 'wkvsfam', 'famvswk', 'hrsrelax', 'secondwk', 'learnnew', 'workfast', 'overwork', 'respect', 'trustman', 'proudemp', 'supcares', 'condemnd', 'promtefr', 'cowrkint', 'jobsecok', 'manvsemp', 'trynewjb', 'health1', 'mntlhlth', 'spvtrfair', 'slpprblm', 'satjob1', 'hyperten', 'stress', 'realinc', 'ballot', 'sei10']

# Define the path to the STATA folder and the new directory to extract files
stata_folder = 'STATA'
unzip_directory = 'Unzipped_GSS'

# Create the new directory if it doesn't exist
if not os.path.exists(unzip_directory):
    os.makedirs(unzip_directory)

# Loop through each compressed zip file in the STATA folder
for filename in os.listdir(stata_folder):
    if filename.endswith('.zip'):
        zip_path = os.path.join(stata_folder, filename)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all .dta files to the new directory
            for file in zip_ref.namelist():
                if file.endswith('.dta'):
                    zip_ref.extract(file, unzip_directory)

print(f"All .dta files have been extracted to the '{unzip_directory}' directory.")

# Read the .dat file with constant spacing between values
dat_file_path = 'GSS.dat'

# Read the .dat file into a pandas DataFrame with whitespace as separator
df = pd.read_csv(dat_file_path, delim_whitespace=True, header=None)

# Assign the labels to the DataFrame columns after reading in the data
if len(df.columns) == len(labels):
    df.columns = labels
else:
    print(f"Warning: The number of columns in the data ({len(df.columns)}) does not match the number of labels ({len(labels)}). Adjusting labels may be required.")

# Display the DataFrame to verify the data
print(df.head())

# Optionally, save the DataFrame to a CSV file for easier use later
df.to_csv('output.csv', index=False)
