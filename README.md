# Hemicellulose ML Optimization

A machine learning pipeline for optimizing hemicellulose extraction conditions using gradient boosting regression and Bayesian optimization.

## Overview

This repository contains code and analysis for predicting and optimizing hemicellulose yield under different extraction conditions. The model is trained on experimental data from Banerjee et al. (2014) and achieves high predictive accuracy (R² = 0.9570) on real-world test samples.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/VedantJadhav701/hemicellulose-ml-optimization.git
cd hemicellulose-ml-optimization

# Install dependencies
pip install -r requirements.txt

# Run the analysis
jupyter notebook notebooks/hemicellulose_optimization_pipeline.ipynb
```

## Repository Structure

```
├── data/
│   ├── raw/                          # Original experimental data
│   └── processed/                    # Processed datasets and outputs
├── notebooks/                        # Jupyter notebooks for analysis
├── figures/                          # Generated visualizations (PNG)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.9570 |
| RMSE | 7.51 mg/g |
| MAE | 6.48 mg/g |
| Max Error | 11.45 mg/g |

## Methodology

The pipeline uses:
- **XGBoost Regression** for yield prediction
- **Feature Engineering** with 16 engineered features (interactions, polynomials, log-transforms)
- **Bayesian Optimization** for finding optimal extraction conditions
- **SHAP Analysis** for feature importance interpretation

## Dataset

The analysis uses the hemicellulose extraction dataset from:
- **Banerjee et al. (2014)** - "Comparative study of 2,3-dibromopropyl acrylate and allyl glycidyl ether modified kraft lignin in unsaturated polyester resin" - Bioresource Technology, 155, pp. 95-102

## Key Findings

### Optimal Extraction Conditions

- **PHWE Method**: 97.19 mg/g hemicellulose yield
- **Alkaline Method**: 157.81 mg/g hemicellulose yield

### Temperature-Time Trade-offs

The model reveals optimal operating windows for both extraction methods with detailed 3D response surface maps provided in the figures/ directory.

## Installation

1. Clone this repository
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Loading the Trained Model

The trained XGBoost model can be loaded and used for predictions:

```python
import joblib

# Load model and scaler
model = joblib.load('hemicellulose_yield_model.pkl')
scaler = joblib.load('hemicellulose_scaler.pkl')
features = joblib.load('hemicellulose_features.pkl')

# Make predictions
X_new = new_data[features]
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

## Reproducibility

This project emphasizes reproducibility:
- ✓ Fixed random seeds for all stochastic operations
- ✓ Exact package versions specified in requirements.txt
- ✓ Complete notebook with all preprocessing and model training steps
- ✓ Validation on strictly physical (non-synthetic) test samples

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{banerjee2014comparative,
  title={Comparative study of 2,3-dibromopropyl acrylate and allyl glycidyl ether modified kraft lignin in unsaturated polyester resin},
  author={Banerjee, A and others},
  journal={Bioresource Technology},
  volume={155},
  pages={95-102},
  year={2014}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Repository**: https://github.com/VedantJadhav701/hemicellulose-ml-optimization
