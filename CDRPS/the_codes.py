import pandas as pd
import numpy as np
import sklearn as sklearn
import xgboost as xgb
import joblib as joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap as shap

#Load dataset
df = pd.read_csv("Road Construction Delay Survey.csv")
df= pd.read_csv("archive(1).csv")

#Show first 5 rows
df.head()
# Shape of dataset
df.shape
# Column names
df.columns
# Summary of data types
df.info()
# Check missing values
df.isnull().sum()



def load_data(df):
    """Load the dataset from a CSV file."""
    return pd.read_csv(df)

#Show first 5 rows
def preview_data(df, n=5):
    """Preview the first n rows of the DataFrame."""
    print(df.head(n))

#validate data
def validate_data(df):
    """Validate the dataset for missing values and correct data types."""

# Check for missing values in the DataFrame.
def check_missing_values(df):
    """Check for missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)

# Get summary of data types and non-null counts.
def info_data(df):
    """Get summary of data types and non-null counts."""
    print(df.info())
    
# Get the shape of the DataFrame.
def shape_data(df):
    """Get the shape of the DataFrame."""
    print(f"Dataset shape: {df.shape}")


#-----------------------------------------------------------------------------#

