# GNN-Assisted Performance Regulation for Photocatalytic NRR on 2D V-GaS

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

This repository implements a **Graph Neural Network (GNN)-assisted framework** for accelerating the screening and analysis of the **Nitrogen Reduction Reaction (NRR)** on sulfur-defected gallium sulfide (2D V-GaS) sheets. The code processes VASP XDATCAR trajectories from molecular dynamics (MD) simulations, builds geometric graphs, and trains models to predict atomic dynamics, identify optimal N2 adsorption sites, and compute attention-based atomic influences.

The work draws from density functional theory (DFT) for material properties and integrates GNN with non-adiabatic molecular dynamics (NAMD) to provide data-driven insights into photocatalytic NRR.

## Features
- **Graph Construction**: Uses PyMatGen's CrystalNN to generate bonded structures from XDATCAR files, incorporating Ga, S, and N2 atoms.
- **GNN Model**: Custom GNN layers with attention mechanisms to model real-time photocarrier dynamics and atomic interactions.
- **Interpretability Tools**: Computes Gibbs free energy profiles, electron-hole recombination times (~4.06 ns), and Ga-S atomic influences on N2 activation.
- **Visualization**: Attention-based bar plots highlighting top influential atoms (e.g., edge Ga sites) using Matplotlib/Seaborn.
- **Screening Acceleration**: GNN-guided identification of optimal N2 binding sites on V-GaS surfaces.

## Installation

1. Install dependencies:
- **PyMatGen**: For crystal structure analysis (`pip install pymatgen`).
- **Torch Geometric**: Follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for GPU support (CUDA-compatible if available).

2. **Data Preparation**: Download or generate your `XDATCAR` file (VASP MD trajectory for 2D V-GaS + N₂ system) and place it in the root directory. Ensure it contains ~5000 timesteps for full training.

## Usage
### 1. Run the Full Pipeline
Execute the main script to load trajectories, train the model, evaluate NRR performance, and visualize atomic influences:
- **Outputs**:
  - Printed training/validation losses and model summary.
  - `best_model.pth`: Saved best GNN weights.
  - `Vaccum-Ga-sheet.png`: Bar plot of Ga-S atom influences on N2 (top sites highlighted in dark red).
- **Expected Runtime**: 10-15 minutes on CPU for 5000 timesteps; faster on GPU.

- Identifies optimal N2 sites via attention scores.

### Example: Optimal Site Identification
After running, inspect the influence plot:
- Top Ga/S atoms (e.g., Ga1, S5) show >3% contribution to N2 activation.
- Use for screening: Higher attention correlates with favorable binding energies.



## Citation
If you use this code or findings in your work, please cite the original paper:

**Title**: Graph Neural Network Assisted Performance Regulation of Photocatalytic Nitrogen Reduction Reaction: An Insight from Machine Learning Accelerated Atomistic Dynamics

**Authors**: Atish Ghosh, Priya Das, Debasis Maji, Debaditya Barman, Pranab Sarkar

**Abstract**: To cross the formidable obstacle in the way of developing renewable energy using photocatalysis needs exact control over the chemical-reactivity of the nanomaterials as well as the behavior of photogenerated charge carriers. We investigated real-time photocarrier dynamics and used graph neural networks (GNN) to accelerate screening of the nitrogen reduction reaction (NRR) mechanism on economical and eco-friendly 2D sulfur-defected gallium sulfide (2D V-GaS). Our density functional theory study revealed its thermal stability, optical properties, and favorable band alignment for NRR. The best site over GaS sheets for the photocatalytic NRR was identified using GNN and MD data. Gibbs free energy calculations showed a downhill energy profile under light-induced potential. A prolonged electron-hole recombination time of 4.06 ns indicates that photogenerated electrons have enough time to reach the active sites of the reaction. Therefore, our study suggests 2D V-GaS is a promising photocatalyst for sustainable and cost-effective NH3 production via NRR.

**Journal Name**:

**DOI**: 
