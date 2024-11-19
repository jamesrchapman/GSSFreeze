import pandas as pd
from sqlalchemy import create_engine

# Load the CSV
df = pd.read_csv("GSS_cumulative_data.csv")

# Create an SQLite database and insert the CSV data
engine = create_engine('sqlite:///GSS_cumulative_data.db')
df.to_sql('GSS', con=engine, index=False, if_exists='replace')
