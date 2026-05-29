# Real-Time Anomaly Detection for Network Intrusion Identification Using Deep Autoencoders

**Course:** INFO-629-001 Applied Artificial Intelligence  
**University:** Drexel University  
**Students:** Madam Kumar Mothkur & Sohan Miryalkar  
**Project:** Real-Time Network Intrusion Detection using Deep Autoencoders

## 1. Project Overview

This repository implements a complete Applied Artificial Intelligence project for detecting anomalous network traffic in real time using an unsupervised deep autoencoder. The system learns normal network behavior from benign traffic and flags suspicious records using reconstruction error.

The implementation follows the project proposal requirements:

- data acquisition and preprocessing
- feature engineering
- deep autoencoder training
- threshold calibration
- baseline comparison against Isolation Forest and One-Class SVM
- real-time replay inference
- explainable feature-level anomaly attribution
- Streamlit operational dashboard
- Dockerized deployment
- reproducible scripts and tests

## 2. System Architecture

```text
Raw / Sample Network Data
        |
        v
Cleaning + Feature Engineering
        |
        v
Encoding + MinMax Scaling
        |
        v
Benign-only Autoencoder Training
        |
        v
Threshold Calibration
        |
        v
Real-Time Replay Inference
        |
        v
Anomaly Event + Feature Attribution
        |
        v
Streamlit Dashboard / Reports
```

## 3. Repository Structure

```text
.
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── explainability.py
│   └── utils.py
├── scripts/
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── simulate_stream.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
└── tests/
```

## 4. Features Implemented

### Deep Autoencoder

Architecture:

```text
Input -> Dense(128) -> Dense(64) -> Bottleneck(32) -> Dense(64) -> Dense(128) -> Output
```

The model is trained only on normal traffic and uses reconstruction error to identify abnormal traffic.

### Threshold Calibration

Thresholds are calibrated from validation-normal reconstruction errors:

- 90th percentile
- 95th percentile
- 99th percentile

The 95th percentile is used as the default operational threshold.

### Baseline Models

The project compares the autoencoder with:

- Isolation Forest
- One-Class SVM

### Real-Time Simulation

The inference engine replays records at configurable speeds such as:

- 100 records/sec
- 500 records/sec
- 1000 records/sec

It records reconstruction error, anomaly decision, latency, and throughput.

### Explainability

Each anomaly is explained using per-feature squared reconstruction error:

```text
e_i = (x_i - x_hat_i)^2
```

The top contributing features are shown to support analyst interpretation.

### Streamlit Dashboard

The dashboard includes:

- anomaly count
- average inference latency
- reconstruction-error timeline
- latency timeline
- anomaly event table
- threshold control
- replay-rate control

## 5. Installation

```bash
git clone https://github.com/sohanmirylkar/INFO-629-001.git
cd INFO-629-001
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## 6. Run Training

```bash
python scripts/run_training.py
```

This creates:

```text
models/autoencoder.keras
models/preprocessor.joblib
models/thresholds.json
```

## 7. Run Evaluation

```bash
python scripts/run_evaluation.py
```

This creates:

```text
reports/evaluation_metrics.json
```

Metrics include:

- precision
- recall
- F1-score
- AUC-ROC
- baseline F1-scores

## 8. Run Stream Simulation

```bash
python scripts/simulate_stream.py
```

## 9. Launch Dashboard

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## 10. Docker Deployment

```bash
docker build -t realtime-anomaly-detector .
docker run -p 8501:8501 realtime-anomaly-detector
```

Then open:

```text
http://localhost:8501
```

## 11. Dataset Notes

The project is designed for NSL-KDD and CICIDS2017-style intrusion detection data. For deadline-safe reproducibility, a deterministic synthetic network-traffic generator is included. This allows the full pipeline to run even if the large benchmark datasets are not present locally.

Optional raw files may be placed in:

```text
data/raw/
```

## 12. Evaluation Framework

The implemented evaluation follows the proposal:

| Metric | Meaning |
|---|---|
| Precision | Proportion of flagged events that are true attacks |
| Recall | Proportion of attacks detected |
| F1-score | Harmonic mean of precision and recall |
| AUC-ROC | Separability of anomaly scores |
| Latency | Time from record to anomaly decision |
| Throughput | Records processed per second |

## 13. Ethical Scope

This project uses public or synthetic benchmark-style network-flow metadata. It does not process personal network traffic or packet payload contents. The system is intended as an analyst-support tool and does not perform autonomous blocking or enforcement.

## 14. Known Limitations

- Synthetic fallback data is not a substitute for full NSL-KDD/CICIDS2017 benchmarking.
- Threshold calibration assumes stable normal traffic distribution.
- Dashboard replay mode simulates live traffic rather than capturing packets from a production network.
- Autoencoder explanations are based on reconstruction error and are not causal explanations.

## 15. Academic Integrity

This repository is a project implementation for academic coursework. Dataset sources, papers, and any external code used in the final submission should be cited properly in the final report.
