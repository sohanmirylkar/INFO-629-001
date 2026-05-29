import json
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger

from src.data_loader import load_dataset
from src.feature_engineering import add_engineered_features
from src.preprocessing import preprocess, split_normal_training
from src.model import build_autoencoder


def train(dataset='nsl-kdd', epochs=50, batch_size=64):
    Path('models').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)

    df = add_engineered_features(load_dataset(dataset=dataset))
    (X_train_full, X_test, y_train_full, y_test), preprocessor, feature_names = preprocess(df)

    X_train_normal, X_val_normal, X_test_normal = split_normal_training(X_train_full, y_train_full)

    model = build_autoencoder(X_train_normal.shape[1])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        CSVLogger('reports/training_log.csv')
    ]

    history = model.fit(
        X_train_normal,
        X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    val_pred = model.predict(X_val_normal, verbose=0)
    val_err = np.mean(np.square(X_val_normal - val_pred), axis=1)
    thresholds = {
        '90': float(np.percentile(val_err, 90)),
        '95': float(np.percentile(val_err, 95)),
        '99': float(np.percentile(val_err, 99))
    }

    model.save('models/autoencoder.keras')
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')

    with open('models/thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    training_summary = {
        'dataset': dataset,
        'epochs_completed': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'thresholds': thresholds,
        'train_normal_records': int(X_train_normal.shape[0]),
        'validation_normal_records': int(X_val_normal.shape[0]),
        'test_records_available': int(X_test.shape[0])
    }

    with open('reports/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)

    return training_summary


if __name__ == '__main__':
    print(json.dumps(train(), indent=2))
