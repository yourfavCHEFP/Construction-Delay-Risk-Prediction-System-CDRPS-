#encode categorical(df)
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def encode_categorical(df):
    """Encode categorical variables in the DataFrame using Label Encoding."""
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders



#scale numerical(df)
from sklearn.preprocessing import StandardScaler

def scale_numerical(df):
    """Scale numerical variables in the DataFrame using Standard Scaling."""
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, scaler



# This completes:

# - Encode categorical variables  
# - Scale numerical features 