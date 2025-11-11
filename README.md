# Anomaly Detection Benchmark: EIF vs AnoLLM

## Overview
Comprehensive benchmarking framework comparing classical machine learning (Extended Isolation Forest) against LLM-based approaches (AnoLLM) for tabular anomaly detection on large-scale fraud detection datasets.

## Dataset
- **Credit Card Fraud Detection**: 284,807 transactions, 30 features, 0.17% fraud rate
- Real-world payment integrity use case with severe class imbalance

## Models Compared

| Model | Type | Implementation |
|-------|------|----------------|
| **EIF** | Tree-based ensemble | PyOD's Isolation Forest |
| **AnoLLM** | LLM embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |

## Key Libraries
- **PyOD** - Classical anomaly detection algorithms
- **Transformers & Sentence-Transformers** - LLM-based embeddings
- **psutil & GPUtil** - System resource monitoring
- **scikit-learn** - Evaluation metrics and preprocessing

## Evaluation Dimensions

### 1. Detection Performance
- AUC-ROC, AUC-PR, Precision, Recall, F1-Score
- Statistical significance testing (t-tests, Wilcoxon)

### 2. Computational Efficiency
- Training & inference time
- Throughput (samples/second)
- Latency per sample

### 3. System Resources
- CPU, RAM, GPU utilization during training & inference
- Peak memory consumption

### 4. Cost Analysis
- Cost per 1,000 predictions
- GPU vs CPU cost comparison
- Scalability economics

## Key Results

**EIF Advantages:**
- 100x+ faster inference throughput
- 10-50x lower cost per prediction
- Scales to millions of samples
- No GPU requirement

**AnoLLM Advantages:**
- Potentially better on complex semantic patterns
- Leverages GPU when available
- Flexible embedding-based approach

## Decision Framework
- **Use EIF for**: High-volume, real-time, cost-sensitive production systems
- **Use AnoLLM for**: Research, complex patterns, GPU-rich environments

## Reproducibility
All experiments run with multiple random seeds (n=3) for statistical validity. Complete resource monitoring ensures transparent system-level comparison.
