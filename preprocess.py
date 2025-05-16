from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def normalize_data(df):
    scaler = StandardScaler()
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    return df
