<p align="center">
  <img src="https://files.manuscdn.com/user_upload_by_module/session_file/310519663329172042/jMpNCKHHMUZhkSGc.png" width="100%" alt="LoRA-Eliminates-Forgetting Banner">
</p>

<h1 align="center">LoRA-Eliminates-Forgetting: Mechanistic Origins of LoRA's Robustness to Catastrophic Forgetting 🧠</h1>

<p align="center">
  <strong>Uncovering why LoRA prevents catastrophic forgetting and enhances generalization.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Research-Paper-red.svg?style=for-the-badge" alt="Research Paper" />
</p>

---

## 🌟 Overview

This repository contains the official code and analysis for our research on **LoRA-Eliminates-Forgetting**. We delve into the mechanistic origins of LoRA's remarkable robustness to catastrophic forgetting in multi-task continual learning settings. Our findings reveal that frozen backbones enforce identical intermediate representations, a key factor in LoRA's superior performance compared to methods like EWC and Full Fine-Tuning.

### 🚀 Key Findings
- **Mechanism Identification:** Frozen backbones in LoRA ensure consistent intermediate representations across tasks.
- **Performance:** LoRA consistently outperforms Elastic Weight Consolidation (EWC) and Full Fine-Tuning in preventing catastrophic forgetting.
- **Generalization:** Enhanced generalization capabilities due to stable feature extraction.

---

## 📂 Repository Structure

```text
├── phase0_profiler.py         # Script for baseline mechanism profiling
├── experiments/               # Directory for experimental setups and results
├── data/                      # Datasets used for multi-task learning
└── requirements.txt           # Project dependencies
```

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/mr-ashish-panday/LoRA-Eliminates-Forgetting.git
cd LoRA-Eliminates-Forgetting
pip install -r requirements.txt
```

### 2. Running Experiments
To reproduce our baseline profiling:
```bash
python phase0_profiler.py
```

Further experimental details and scripts can be found in the `experiments/` directory.

---

## ⚖️ License
This project is licensed under the **MIT License**.

## 🤝 Contributing
We welcome contributions to this research! Please refer to our contribution guidelines for more details.
