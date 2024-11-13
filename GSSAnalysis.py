import json

# Replace 'your_file.json' with the path to your JSON file
file_path = 'GSS.json'


with open(file_path, 'r') as file:
    # Read the initial chunk of data (adjust size if necessary)
    snippet = file.read(10000)  # This reads only a portion to find the initial keys

print(snippet)
# Manually search for top-level keys using regular expressions
import re

# Regular expression to match JSON-style keys in dictionaries
pattern = r'"([^"]+)":'

# Find all matches, which correspond to keys in the JSON dictionary
keys = re.findall(pattern, snippet)

# Remove duplicates in case some keys repeat within the snippet
unique_keys = list(dict.fromkeys(keys))

# Save these keys to a .dat file if they look reasonable
if unique_keys:
    with open('labels.dat', 'w') as label_file:
        label_file.write("\n".join(unique_keys))
    print("Extracted labels saved to labels.dat:", unique_keys)
else:
    print("No keys found, or snippet was too short to capture structure.")