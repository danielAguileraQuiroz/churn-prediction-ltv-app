import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare_data(df: pd.DataFrame):
    df = df.copy()
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
    if 'Tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['LTV_Proxy'] = df['Tenure'] * df['MonthlyCharges']
    elif 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['LTV_Proxy'] = df['tenure'] * df['MonthlyCharges']
    else:
        df['LTV_Proxy'] = df['TotalCharges']
        
    return df

def get_preprocessor(categorical_cols, numerical_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
