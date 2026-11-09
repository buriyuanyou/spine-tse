# Execution is a Proxy for Knowledge: An EFSM-Guided Memory Unit Framework for Code Optimization

This repository contains the official implementation of our framework for automated code optimization.

## 🚀 Quick Start

### ⚙️ Preparation

#### 1. Environment Setup

Create the required Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate spine
```

#### 2. External Data

The `processed_data` required for this project is provided by the official [SBLLM repository](https://github.com/shuzhenggao/sbllm). Please download the folder from the [Zenodo records](https://zenodo.org/records/14096664) and place its contents into the currently empty `./processed_data` directory.

#### 3. Dataset Acquisition

This project utilizes the **PIE-Perf** dataset. Please download the Python and C++ splits from the official sources:

- **Python Dataset:** [Download Link](https://drive.google.com/file/d/1ec8eOWgnBrzy2HlNDlTX6iURwQcIxDXH/view?usp=sharing)
- **C++ Dataset:** [Download Link](https://drive.google.com/file/d/1NqMT7kqCwk99hj4BjpUcsxLIzPFv_DtT/view?usp=sharing)
- **Reference:** [PIE-Perf GitHub Repository](https://github.com/madaan/pie-perf)

---

## 💻 Workflow

### 1. Pattern Generation

#### Phase A: Data Simplification

Before generating patterns, simplify the raw training data. Modify the `input_path` attribute in `simplify_data.py` to point to your downloaded PIE-Perf files:

- `path/to/python_splits/train.jsonl`
- `path/to/cpp_splits/train.jsonl`

Run the script:

```bash
python simplify_data.py
```

#### Phase B: Mining Optimization Patterns

Use the simplified training data to extract optimization patterns.

Execute the generation scripts:

```bash
python generate_patterns_python.py
python generate_patterns_cpp.py
```

---

### 2. Code Optimization (Inference)

Execute the optimization pipeline:

```bash
bash run.sh
```
