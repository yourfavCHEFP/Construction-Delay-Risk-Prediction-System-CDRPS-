#split dataset(df, target_column, test_size, random_state)
from sklearn.model_selection import train_test_split

def split_dataset(df, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test







# This completes:
# - Split dataset into train/val/test sets