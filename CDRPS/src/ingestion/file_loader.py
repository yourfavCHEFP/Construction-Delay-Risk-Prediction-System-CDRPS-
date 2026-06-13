# This file will handle:

# - Uploading CSV/Excel  
# - Validating file format 
# - Loading data into a DataFrame 
# - Previewing first rows  

# - Responsibility 1: Accept a file path from the user (CSV or Excel)  
# - Responsibility 2: Check that the file extension is one of: `.csv`, `.xlsx`, `.xls`  
# - Responsibility 3: If extension is invalid → raise a clear, human‑readable error message  
# - Responsibility 4: If valid → load the file into a pandas DataFrame  
# - Responsibility 5: Provide a way to preview the first N rows (default 5)  
# - Responsibility 6: Never print raw stack traces to the user—only clean messages  


import pandas as pd

# 3. validate_file_extension(path: str)
def validate_file_extension(path: str):
    """Validate the file extension."""
    if not (path.endswith('.csv') or path.endswith('.xlsx')):
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
#load file
def load_file(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file after validating the file extension."""
    validate_file_extension(path)
    return pd.read_csv(path)


# 2. preview_data(df: pd.DataFrame, n=5)
def preview_data(df: pd.DataFrame, n=5):
    """Preview the first n rows of the DataFrame."""
    print(df.head(n))