import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_data(df: pd.DataFrame):
    """
    Limpia datos, convierte tipos de datos y calcula una métrica proxy de LTV.
    """
    df = df.copy()
    
    # Limpieza de nulos o espacios en blanco en TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
    # Estimación del LTV (Tenure * MonthlyCharges)
    if 'Tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['LTV_Proxy'] = df['Tenure'] * df['MonthlyCharges']
    elif 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['LTV_Proxy'] = df['tenure'] * df['MonthlyCharges']
    else:
        df['LTV_Proxy'] = df['TotalCharges']
        
    return df

def get_preprocessor(categorical_cols, numerical_cols):
    """
    Retorna un ColumnTransformer listo para encadenar en Scikit-Learn.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )