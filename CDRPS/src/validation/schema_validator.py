# - Responsibility 1: Verify that all required columns exist in the dataset  
# - Responsibility 2: Report which required columns are missing  
# - Responsibility 3: Detect missing values in required fields  
# - Responsibility 4: Provide a summary of missing values per column  
# - Responsibility 5: Detect obvious invalid values (e.g., negative durations, impossible rainfall)  
# - Responsibility 6: Return structured results (not just print), so the UI or notebook can decide how to show them



#check_required_columns(df, required_columns)
def check_required_columns(df, required_columns):
    """Check if the required columns are present in the DataFrame."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        print("All required columns are present.")


#detect missing values(df)
def check_missing_values(df):
    """Check for missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)


#detect_inconsistent_values(df)
def detect_inconsistent_values(df):
    """Detect inconsistent values in the DataFrame."""
    for column in df.columns:
        unique_values = df[column].unique()
        print(f"Unique values in column '{column}': {unique_values}")