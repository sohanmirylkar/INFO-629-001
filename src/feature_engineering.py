import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {'src_bytes','dst_bytes'}.issubset(df.columns):
        df['byte_asymmetry'] = (df['src_bytes'] - df['dst_bytes']).abs() / (df['src_bytes'] + df['dst_bytes'] + 1e-9)
        df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
    if {'count','srv_count'}.issubset(df.columns):
        df['connection_ratio'] = df['count'] / (df['srv_count'] + 1e-9)
    numeric_cols = [c for c in df.columns if c != 'label' and pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols[:4]:
        df[f'{col}_rolling_mean_5'] = df[col].rolling(5, min_periods=1).mean()
        df[f'{col}_rolling_std_5'] = df[col].rolling(5, min_periods=1).std().fillna(0)
    return df
