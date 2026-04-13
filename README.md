# Hemicellulose ML Optimization

Machine learning optimization analysis for pressurized hot-water extraction (PHWE) of hemicellulose from sugarcane bagasse.

## Project Overview

This project applies Bayesian Optimization and response surface modeling to optimize hemicellulose yield extraction under various process conditions (temperature, time, liquid-to-solid ratio).

## Dataset

- **Source**: Expanded hemicellulose extraction dataset
- **Size**: 200 interpolated data points
- **Features**: Temperature, Time, LSR (Liquid-to-Solid Ratio)
- **Target**: Hemicellulose Yield

## Contents

### `/data`
- `raw/`: Original dataset (hemicellulose_200pts.csv)
- `proccessed/`: Bayesian optimization results (EI and UCB acquisition functions)

### `/figures`
Visualization outputs including:
- Surface plots and contour maps
- Bayesian optimization convergence analysis
- GP uncertainty landscapes
- Acquisition function landscapes
- Feature importance and sensitivity analysis

### `/notebooks`
- `hemicellulose_bayesian_optimization.ipynb`: Main analysis notebook with Gaussian Process modeling, Bayesian optimization, and result visualization

## Methods

- **Model**: Gaussian Process Regression (scikit-learn)
- **Optimization**: Bayesian Optimization with multiple acquisition functions (EI, UCB)
- **Visualization**: 3D surface plots, contour maps, convergence curves

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
scipy
```

## Results

Bayesian optimization identified optimal extraction conditions to maximize hemicellulose yield. Results saved in CSV format with both Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition strategies.

---

**Author**: Vedant Jadhav  
**Repository**: https://github.com/VedantJadhav701/hemicellulose-ml-optimization
