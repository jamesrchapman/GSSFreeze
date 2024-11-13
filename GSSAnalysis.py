import json

# Replace 'your_file.json' with the path to your JSON file
file_path = 'GSS.json'


# Read the file as a string and replace single quotes with double quotes
with open(file_path, 'r') as file:
    raw_data = file.read().replace("'", '"')  # Replaces single quotes with double quotes

# Load the corrected JSON data
try:
    data = json.loads(raw_data)
    print("Top-level keys in the JSON data:", data.keys())
except json.JSONDecodeError as e:
    print("Error decoding JSON:", e)