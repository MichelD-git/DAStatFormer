# DAStatFormer  
**A Hybrid Multibranch Transformer with Statistical Features for DAS-Based Event Recognition**

---

## Overview

This repository contains the official PyTorch implementation of **DAStatFormer**, a hybrid Transformer-based framework for **event recognition from Distributed Acoustic Sensing (DAS) data**.

DAStatFormer combines:
- **Compact multidomain statistical features** (temporal, waveform, spectral),
- **Multibranch Gated Transformer Networks (GTN)**,
- **Step-wise and channel-wise self-attention with adaptive gating**,

to achieve **accurate, scalable, and interpretable classification** of DAS events while drastically reducing computational cost compared to raw-signal-based approaches.

This code accompanies a paper submitted to **ICPR 2026**.

---

## Key Contributions

- Statistical feature-driven Transformer for DAS, avoiding raw high-dimensional signal processing  
- 24 ANOVA-selected interpretable features per channel, reducing data size by orders of magnitude  
- Multibranch attention architecture modeling both spatial (channel-wise) and attribute-wise dependencies  
- Efficient and scalable design suitable for real-time DAS monitoring  
- Validation on laboratory and real-world DAS datasets  

---

## Data

This repository relies on publicly available DAS datasets.  
**Raw data are not included** and must be downloaded separately.

### Dataset 1 — Laboratory φ-OTDR Dataset
- Source: Xiaomin Cao et al. Beijing Jiaotong University (BJTUSensor)
- Link: https://github.com/BJTUSensor/
- Description: Controlled laboratory φ-OTDR experiments with **6 event classes** (standard intrusion events) .
  - background
  - digging
  - knocking
  - shaking
  - watering
  - walking
   

### Dataset 2 — Real-Scenario Campus DAS Dataset
- Source: Tomasov *et al.* (Figshare)
- DOI: https://doi.org/10.6084/m9.figshare.27004732
- Description: Real-world φ-OTDR dataset with **9 event classes**:
  - car
  - fence
  - longboard
  - manipulation
  - open / close
  - construction
  - regular
  - running
  - walking


## Repository Structure

```text
DAStatFormer/
├── dataset_process/
│   └── dataset_process.py     # Dataset loading and preprocessing
├── module/
│   ├── encoder.py             # Transformer encoder (MHA + FFN)
│   ├── attention.py           # Multi-head attention modules
│   └── loss.py                # Loss functions
├── utils/
│   ├── random_seed.py         # Reproducibility utilities
│   └── visualization.py       # Training curves & confusion matrices
├── feature_extraction.py      # 24 statistical feature extraction
├── DAStatFormer.py                   # Main training and evaluation script
├── requirements.txt
└── README.md
