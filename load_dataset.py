import pandas as pd

# Load the dataset
data = pd.read_csv("german_credit_data.csv")

# Preview the dataset
print(data.head())

# Check for missing values and data types
print(data.info())
print(data.isnull().sum())
