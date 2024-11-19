from sqlalchemy import create_engine
import pandas as pd

# Connect to the existing SQLite database
engine = create_engine('sqlite:///GSS_cumulative_data.db')

# Load the table into a DataFrame
df_loaded = pd.read_sql('GSS', con=engine)

# Check the loaded data
print(df_loaded.head())

