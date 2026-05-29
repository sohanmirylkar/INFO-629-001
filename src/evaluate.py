import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import OneClassSVM

from src.data_loader import load_dataset
from src.feature_engineering import add_engineered_features
from src.preprocessing import preprocess
from src.pca_analysis import run_pca


def _metrics(y_true, y_pred, scores=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'false_positive_rate': float(fp / (fp + tn + 1e-12)),
        'true_positive': int(tp),
        'false_positive': int(fp),
        'true_negative': int(tn),
        'false_negative': int(fn)
    }
    if scores is not None:
        result['auc_roc'] = float(roc_auc_score(y_true, scores))
    return result


def evaluate(dataset='nsl-kdd'):
    Path('reports').mkdir(exist_ok=True)
    df = add_engineered_features(load_dataset(dataset=dataset))
    (X_train, X_test, y_train, y_test), _, _ = preprocess(df)

    run_pca(X_train, output_dir='reports')

    model = tf.keras.models.load_model('models/autoencoder.keras')
    reconstructed = model.predict(X_test, verbose=0)
    reconstruction_error = np.mean(np.square(X_test - reconstructed), axis=1)

    try:
        thresholds = json.load(open('models/thresholds.json'))
    except FileNotFoundError:
        thresholds = {'90': float(np.percentile(reconstruction_error, 90)), '95': float(np.percentile(reconstruction_error, 95)), '99': float(np.percentile(reconstruction_error, 99))}

    threshold_analysis = {}
    for percentile, threshold in thresholds.items():
        y_pred = (reconstruction_error > threshold).astype(int)
        threshold_analysis[percentile] = _metrics(y_test, y_pred, reconstruction_error)

    isolation_forest = IsolationForest(contamination=0.10, random_state=42)
    isolation_forest.fit(X_train)
    iso_scores = -isolation_forest.decision_function(X_test)
    iso_pred = (isolation_forest.predict(X_test) == -1).astype(int)

    one_class_svm = OneClassSVM(nu=0.10, kernel='rbf', gamma='scale')
    one_class_svm.fit(X_train)
    svm_scores = -one_class_svm.decision_function(X_test)
    svm_pred = (one_class_svm.predict(X_test) == -1).astype(int)

    metrics = {
        'autoencoder_95': threshold_analysis.get('95', next(iter(threshold_analysis.values()))),
        'isolation_forest': _metrics(y_test, iso_pred, iso_scores),
        'one_class_svm': _metrics(y_test, svm_pred, svm_scores),
        'threshold_analysis': threshold_analysis
    }

    comparison_rows = []
    for model_name in ['autoencoder_95', 'isolation_forest', 'one_class_svm']:
        row = {'model': model_name}
        row.update(metrics[model_name])
        comparison_rows.append(row)

    pd.DataFrame(comparison_rows).to_csv('reports/model_comparison.csv', index=False)
    with open('reports/threshold_analysis.json', 'w') as f:
        json.dump(threshold_analysis, f, indent=2)
    with open('reports/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
