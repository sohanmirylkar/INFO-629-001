import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def build_preprocessor(X: pd.DataFrame):
    cat = [c for c in X.columns if X[c].dtype == 'object']
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ('num', MinMaxScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)
    ]), num, cat


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    names = list(numeric_cols)
    if categorical_cols:
        encoder = preprocessor.named_transformers_['cat']
        names.extend(encoder.get_feature_names_out(categorical_cols).tolist())
    return names


def preprocess(df: pd.DataFrame, label_col: str = 'label'):
    df = df.drop_duplicates().replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    pre, num, cat = build_preprocessor(X)
    Xp = pre.fit_transform(X)
    feature_names = get_feature_names(pre, num, cat)
    return train_test_split(Xp, y, test_size=0.2, stratify=y, random_state=42), pre, feature_names


def split_normal_training(X, y):
    X_normal = X[y == 0]
    X_train, X_temp = train_test_split(X_normal, test_size=0.30, random_state=42)
    X_val, X_test_normal = train_test_split(X_temp, test_size=0.50, random_state=42)
    return X_train, X_val, X_test_normal
