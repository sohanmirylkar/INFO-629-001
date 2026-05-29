import numpy as np
import pandas as pd


def feature_attribution(original, reconstructed, feature_names=None, top_k=10):
    errors=np.square(np.asarray(original)-np.asarray(reconstructed)).ravel()
    total=errors.sum()+1e-12
    scores=errors/total
    if feature_names is None:
        feature_names=[f'feature_{i}' for i in range(len(scores))]
    df=pd.DataFrame({'feature':feature_names,'attribution':scores,'squared_error':errors})
    return df.sort_values('attribution',ascending=False).head(top_k)


def severity_score(error, threshold):
    if threshold<=0: return 0.0
    return float(min(100,(error/threshold)*50))
