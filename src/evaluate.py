import json, numpy as np, tensorflow as tf
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from src.data_loader import load_dataset
from src.feature_engineering import add_engineered_features
from src.preprocessing import preprocess


def evaluate():
    df=add_engineered_features(load_dataset())
    (X_train,X_test,y_train,y_test),_=preprocess(df)
    model=tf.keras.models.load_model('models/autoencoder.keras')
    pred=model.predict(X_test,verbose=0)
    err=np.mean(np.square(X_test-pred),axis=1)
    threshold=np.percentile(err,95)
    ae=(err>threshold).astype(int)

    iso=IsolationForest(contamination=0.1,random_state=42).fit(X_train)
    iso_pred=(iso.predict(X_test)==-1).astype(int)

    svm=OneClassSVM(nu=0.1).fit(X_train)
    svm_pred=(svm.predict(X_test)==-1).astype(int)

    metrics={
      'autoencoder':{'precision':float(precision_score(y_test,ae)),'recall':float(recall_score(y_test,ae)),'f1':float(f1_score(y_test,ae)),'auc':float(roc_auc_score(y_test,err))},
      'isolation_forest':{'f1':float(f1_score(y_test,iso_pred))},
      'one_class_svm':{'f1':float(f1_score(y_test,svm_pred))}
    }
    with open('reports/evaluation_metrics.json','w') as f: json.dump(metrics,f,indent=2)
    return metrics
