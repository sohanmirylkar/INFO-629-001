import json,time,joblib,numpy as np,pandas as pd,tensorflow as tf
from src.data_loader import load_dataset
from src.feature_engineering import add_engineered_features


def stream(rate=100,limit=500):
    model=tf.keras.models.load_model('models/autoencoder.keras')
    pre=joblib.load('models/preprocessor.joblib')
    threshold=json.load(open('models/thresholds.json'))['95']
    df=add_engineered_features(load_dataset()).head(limit)
    y=df['label'] if 'label' in df.columns else None
    X=df.drop(columns=['label'])
    X=pre.transform(X)
    events=[]
    for idx,row in enumerate(X):
        start=time.time()
        pred=model.predict(row.reshape(1,-1),verbose=0)
        err=float(np.mean(np.square(row-pred[0])))
        latency=(time.time()-start)*1000
        events.append({'record':idx,'error':err,'anomaly':int(err>threshold),'latency_ms':latency})
        time.sleep(max(0,1/rate))
    return pd.DataFrame(events)
