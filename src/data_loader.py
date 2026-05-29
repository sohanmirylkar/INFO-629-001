import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n=5000):
    rng=np.random.default_rng(42)
    df=pd.DataFrame({
        'duration':rng.normal(10,3,n),
        'src_bytes':rng.normal(500,100,n),
        'dst_bytes':rng.normal(300,80,n),
        'count':rng.integers(1,50,n),
        'srv_count':rng.integers(1,30,n),
        'protocol_type':rng.choice(['tcp','udp','icmp'],n),
        'label':rng.choice([0,1],n,p=[0.9,0.1])
    })
    return df


def load_dataset(path=None):
    if path and Path(path).exists():
        return pd.read_csv(path)
    return generate_sample_data()
