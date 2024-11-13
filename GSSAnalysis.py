import json
import pandas as pd

# Replace 'your_file.json' with the path to your JSON file
file_path = 'GSS.json'
labels=['year', 'id_', 'hrs1', 'hrs2', 'wrkslf', 'occ10', 'sphrs1', 'sphrs2', 'happy', 'hapmar', 'joblose', 'satjob', 'class_', 'satfin', 'finalter', 'tvhours', 'wrktype', 'yearsjob', 'waypaid', 'wrksched', 'moredays', 'mustwork', 'wrkhome', 'whywkhme', 'famwkoff', 'wkvsfam', 'famvswk', 'hrsrelax', 'secondwk', 'learnnew', 'workfast', 'overwork', 'respect', 'trustman', 'proudemp', 'supcares', 'condemnd', 'promtefr', 'cowrkint', 'jobsecok', 'manvsemp', 'trynewjb', 'health1', 'mntlhlth', 'spvtrfair', 'slpprblm', 'satjob1', 'hyperten', 'stress', 'realinc', 'ballot', 'sei10']
dat_file_path = 'GSS.dat'


# Read the .dat file into a pandas DataFrame with whitespace as separator
df = pd.read_csv(dat_file_path, delim_whitespace=True, names=labels)

# Display the DataFrame to verify the data
print(df.head())

# Optionally, save the DataFrame to a CSV file for easier use later
df.to_csv('output.csv', index=False)
