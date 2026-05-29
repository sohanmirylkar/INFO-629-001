from pathlib import Path
import numpy as np
import pandas as pd

NSL_KDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
    'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack_type','difficulty'
]


def _binary_label(value):
    text = str(value).strip().lower().replace('.', '')
    return 0 if text in {'normal', 'benign', '0'} else 1


def generate_sample_data(n=6000, anomaly_ratio=0.12):
    rng = np.random.default_rng(42)
    y = rng.choice([0, 1], n, p=[1 - anomaly_ratio, anomaly_ratio])
    duration = rng.gamma(2.0, 3.0, n) + y * rng.gamma(2.0, 8.0, n)
    src_bytes = rng.normal(500, 120, n) + y * rng.normal(900, 350, n)
    dst_bytes = rng.normal(300, 90, n) + y * rng.normal(80, 50, n)
    df = pd.DataFrame({
        'duration': np.clip(duration, 0, None),
        'protocol_type': rng.choice(['tcp', 'udp', 'icmp'], n, p=[0.72, 0.20, 0.08]),
        'service': rng.choice(['http', 'smtp', 'ftp_data', 'private', 'domain_u'], n),
        'flag': rng.choice(['SF', 'S0', 'REJ', 'RSTR'], n),
        'src_bytes': np.clip(src_bytes, 0, None),
        'dst_bytes': np.clip(dst_bytes, 0, None),
        'logged_in': rng.choice([0, 1], n, p=[0.35, 0.65]),
        'count': rng.integers(1, 60, n) + y * rng.integers(20, 120, n),
        'srv_count': rng.integers(1, 40, n),
        'serror_rate': np.clip(rng.beta(1, 8, n) + y * rng.beta(2, 3, n), 0, 1),
        'same_srv_rate': np.clip(rng.beta(7, 2, n) - y * rng.beta(2, 7, n), 0, 1),
        'dst_host_count': rng.integers(1, 255, n),
        'dst_host_srv_count': rng.integers(1, 255, n),
        'label': y.astype(int)
    })
    return df


def load_nsl_kdd(path='data/raw/KDDTrain+.txt'):
    path = Path(path)
    if not path.exists():
        return generate_sample_data()
    df = pd.read_csv(path, names=NSL_KDD_COLUMNS)
    df['label'] = df['attack_type'].apply(_binary_label)
    return df.drop(columns=['attack_type', 'difficulty'], errors='ignore')


def load_cicids2017(path='data/raw/CICIDS2017.csv'):
    path = Path(path)
    if not path.exists():
        return generate_sample_data()
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(' ', '_').replace('/', '_') for c in df.columns]
    label_col = next((c for c in df.columns if c.lower() == 'label'), None)
    if label_col is None:
        raise ValueError('CICIDS2017 file must contain a Label column.')
    df['label'] = df[label_col].apply(_binary_label)
    df = df.drop(columns=[label_col], errors='ignore')
    object_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'label']
    for col in object_cols:
        if df[col].nunique() > 50:
            df = df.drop(columns=[col])
    return df


def load_dataset(path=None, dataset='nsl-kdd'):
    if path and Path(path).exists():
        if str(path).endswith('.txt'):
            return load_nsl_kdd(path)
        return pd.read_csv(path)
    if dataset.lower() == 'cicids2017':
        return load_cicids2017()
    return load_nsl_kdd()
