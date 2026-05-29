import json, joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset
from src.feature_engineering import add_engineered_features
from src.preprocessing import preprocess
from src.model import build_autoencoder


def train():
    df=add_engineered_features(load_dataset())
    (X_train,X_test,y_train,y_test),pre=preprocess(df)
    X_normal=X_train[y_train==0]
    model=build_autoencoder(X_normal.shape[1])
    model.fit(X_normal,X_normal,epochs=20,batch_size=64,validation_split=0.2,verbose=0)
    pred=model.predict(X_normal,verbose=0)
    err=np.mean(np.square(X_normal-pred),axis=1)
    thresholds={'90':float(np.percentile(err,90)),'95':float(np.percentile(err,95)),'99':float(np.percentile(err,99))}
    Path('models').mkdir(exist_ok=True)
    model.save('models/autoencoder.keras')
    joblib.dump(pre,'models/preprocessor.joblib')
    with open('models/thresholds.json','w') as f: json.dump(thresholds,f)
    return thresholds

if __name__=='__main__':
    print(train())
