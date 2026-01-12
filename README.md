# Hybrid IDS: XGBoost + SHAP + LLM for Edge-Deployable Intrusion Detection

A hybrid **Intrusion Detection System** combining XGBoost classification, SHAP explainability, and quantized LLM narrative generation — optimized for edge deployment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.42+-orange.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

## Abstract

This project implements a **5-phase hybrid intrusion detection pipeline**:

1. **Phase 1:** Data preprocessing with correlation-based feature selection
2. **Phase 2:** XGBoost detection engine with Optuna hyperparameter optimization
3. **Phase 3:** SHAP-based global and local interpretability
4. **Phase 4:** 4-bit quantized LLM for human-readable security narratives
5. **Phase 5:** Edge feasibility analysis with latency and resource profiling

**Key Contribution:** Complete offline IDS pipeline achieving **~90% accuracy** with **sub-millisecond detection latency** and **<8GB VRAM** for the full stack.

---

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Network Flow   │ -> │    XGBoost      │ -> │      SHAP       │ -> │  Qwen2.5-1.5B   │
│   (34 features) │    │   Classifier    │    │  TreeExplainer  │    │  (4-bit quant)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
   UNSW-NB15              Binary + Conf          Top-k Features        Security Report
    Dataset               (Attack/Normal)        + SHAP Values         (150 words)
```

---

## Performance Summary

### Detection Engine Comparison (Phase 2)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Inference Speed |
|-------|----------|-----------|--------|----------|---------|-----------------|
| **XGBoost (Baseline)** | 90.11% | 98.68% | 86.63% | 92.26% | 98.53% | **321,884/sec** |
| XGBoost (Optuna) | 89.65% | 98.97% | 85.68% | 91.85% | 98.61% | 101,845/sec |
| Random Forest | 90.12% | 98.89% | 86.45% | 92.25% | 98.63% | 57,059/sec |
| DNN (3-layer) | 91.54% | 97.03% | 90.34% | 93.56% | 97.91% | 29,576/sec |

### Pipeline Latency (Phase 5)

| Component | Latency | Target | Status |
|-----------|---------|--------|--------|
| XGBoost Detection | ~0.003 ms/sample | < 10 ms | ✅ |
| SHAP Explanation | ~10-50 ms/sample | < 500 ms | ✅ |
| LLM Generation | ~10-13 sec/narrative | < 30 sec | ✅ |

### Resource Requirements

| Resource | Peak Usage | Edge Target |
|----------|------------|-------------|
| System RAM | < 8 GB | < 16 GB |
| GPU VRAM | ~3-4 GB | < 8 GB |
| Model Size | ~1.5 GB (quantized) | < 4 GB |

---

## Quick Start

### Prerequisites

- **Python** 3.8+
- **CUDA-capable GPU** (4GB+ VRAM recommended)
- **RAM** 16GB (8GB minimum)
- **Git LFS** for dataset files

### Installation

```bash
# Clone repository
git clone https://github.com/Arif-Foysal/hybrid-ids-xboost-and-llm.git
cd hybrid-ids-xboost-and-llm

# Install Git LFS and pull large files
git lfs install
git lfs pull

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Option 1: Jupyter Notebook
jupyter notebook ids.ipynb

# Option 2: VS Code (with Jupyter extension)
code ids.ipynb

# Option 3: JupyterLab
jupyter lab ids.ipynb
```

**Execution Order:** Run cells sequentially from Phase 1 → Phase 5. Each phase depends on the previous.

---

## Project Structure

```
hybrid-ids-xboost-and-llm/
│
├── ids.ipynb                    # Main notebook (95 cells, 5 phases)
├── phases.md                    # Phase documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── README.md                    # This file
│
├── CSV Files/                   # UNSW-NB15 Dataset (Git LFS)
│   ├── UNSW-NB15_1.csv         # Raw data part 1 (161 MB)
│   ├── UNSW-NB15_2.csv         # Raw data part 2 (157 MB)
│   ├── UNSW-NB15_3.csv         # Raw data part 3 (147 MB)
│   ├── UNSW-NB15_4.csv         # Raw data part 4 (93 MB)
│   ├── NUSW-NB15_features.csv  # Feature descriptions
│   ├── NUSW-NB15_GT.csv        # Ground truth labels
│   └── Training and Testing Sets/
│       ├── UNSW_NB15_training-set.csv  # 82,332 samples
│       └── UNSW_NB15_testing-set.csv   # 175,341 samples
│
└── outputs/                     # Generated during execution
    ├── correlation_matrix.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── shap_feature_importance.png
    ├── shap_beeswarm.png
    ├── narrative_generation_analysis.png
    ├── edge_feasibility_analysis.png
    ├── xgboost_model.json
    ├── trained_models.pkl
    └── *.csv                    # Various result tables
```

---

## Notebook Structure (ids.ipynb)

### Phase 1: Data Curation & Preprocessing (Cells 1-22)

| Step | Description | Key Output |
|------|-------------|------------|
| 1.1 | Load UNSW-NB15 train/test CSVs | 257,673 total samples |
| 1.2 | Handle missing values (`-` → `none`) | Clean dataset |
| 1.3 | Label encode categoricals (`proto`, `service`, `state`) | Numeric features |
| 1.4 | Correlation analysis (threshold=0.95) | 34 features (from 44) |

**Features Removed:** `dloss`, `sloss`, `ct_ftp_cmd`, `dbytes`, `sbytes`, `dwin`, `ct_srv_dst`, `ct_src_dport_ltm`

### Phase 2: Detection Engine (Cells 23-43)

| Step | Description | Key Output |
|------|-------------|------------|
| 2.1 | Train baseline XGBoost | 90.11% accuracy |
| 2.2 | Optuna HPO (50 trials) | Best params JSON |
| 2.3 | Train Random Forest + DNN | Comparison table |
| 2.4 | Generate visualizations | Confusion matrices, ROC curves |

**Optuna Search Space:**
```python
{
    'max_depth': [3, 12],
    'learning_rate': [0.01, 0.3],
    'n_estimators': [100, 1000],
    'min_child_weight': [1, 10],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0]
}
```

### Phase 3: Interpretability Layer (Cells 44-56)

| Step | Description | Key Output |
|------|-------------|------------|
| 3.1 | SHAP TreeExplainer setup | 5,000 sample analysis |
| 3.2 | Global feature importance | Bar plot + CSV |
| 3.3 | Beeswarm plot | Feature impact direction |
| 3.4 | Local explanations | Per-sample breakdown |

**Top 5 Features (by mean |SHAP|):**
1. `sttl` - Source TTL
2. `ct_state_ttl` - Connection state TTL
3. `sload` - Source bits per second
4. `ct_dst_src_ltm` - Connection count
5. `smean` - Mean packet size (src)

### Phase 4: Generative Narrative (Cells 57-77)

| Step | Description | Key Output |
|------|-------------|------------|
| 4.1 | Load Qwen2.5-1.5B-Instruct (4-bit) | ~3GB VRAM |
| 4.2 | Dynamic prompt template | Security-focused prompts |
| 4.3 | Batch generation (50 attacks) | Narrative CSV |
| 4.4 | Raw vs. generated comparison | Before/after table |

**LLM Configuration:**
```python
{
    'model': 'Qwen/Qwen2.5-1.5B-Instruct',
    'quantization': '4-bit NF4',
    'max_new_tokens': 200,
    'temperature': 0.7,
    'top_p': 0.9
}
```

### Phase 5: Edge Feasibility Analysis (Cells 78-95)

| Step | Description | Key Output |
|------|-------------|------------|
| 5.1 | Latency benchmarks | Per-component timing |
| 5.2 | Resource monitoring | CPU/RAM/VRAM profiles |
| 5.3 | Scalability analysis | Throughput vs batch size |
| 5.4 | Deployment recommendations | Edge device compatibility |

---

## Technical Details

### Dataset: UNSW-NB15

- **Source:** University of New South Wales
- **Total Records:** 2,540,044 (raw), 257,673 (train/test split)
- **Features:** 49 (raw), 34 (after preprocessing)
- **Attack Categories:** 9 types + Normal
  - Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

### Model Specifications

**XGBoost (Recommended for Edge):**
```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    random_state=42
)
```

**SHAP Explainer:**
```python
shap.TreeExplainer(model)  # Optimized for tree-based models
# Computation: ~10ms per sample
```

**Quantized LLM:**
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

---

## Generated Outputs

After running all phases, the following files are created:

| File | Phase | Description |
|------|-------|-------------|
| `correlation_matrix.png` | 1 | Feature correlation heatmap |
| `preprocessed_data.pkl` | 1 | Cleaned dataset pickle |
| `model_comparison.png` | 2 | Model performance visualization |
| `confusion_matrices.png` | 2 | Per-model confusion matrices |
| `xgboost_model.json` | 2 | Trained XGBoost model |
| `shap_feature_importance.png` | 3 | Global SHAP importance |
| `shap_beeswarm.png` | 3 | Feature impact beeswarm |
| `narrative_generation_results.csv` | 4 | Generated narratives |
| `edge_feasibility_metrics.json` | 5 | Performance metrics |
| `edge_feasibility_table.csv` | 5 | Paper-ready metrics |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and test
4. Commit: `git commit -m "Add new feature"`
5. Push: `git push origin feature/new-feature`
6. Open a Pull Request

### Development Notes

- Run all 5 phases sequentially before testing changes
- Use `%pip install` in notebooks for Colab compatibility
- Test on both GPU and CPU configurations
- Update this README if adding new phases/outputs

---

## Citation

If you use this work, please cite:

```bibtex
@software{hybrid_ids_xgboost_llm,
  author = {Foysal, Arif},
  title = {Hybrid IDS: XGBoost + SHAP + LLM for Edge Deployment},
  year = {2026},
  url = {https://github.com/Arif-Foysal/hybrid-ids-xboost-and-llm}
}
```

**Dataset Citation:**
```bibtex
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: A comprehensive data set for network intrusion detection systems},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={MilCIS},
  year={2015}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Arif Faysal** - [GitHub](https://github.com/Arif-Foysal)

---

⭐ **Star this repo** if you find it useful for your research!
