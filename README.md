# Protein Tertiary Structure Regression Project

This project aims to predict the RMSD (Root Mean Square Deviation) of protein residues using various structural and physicochemical features. The workflow follows standard data science steps:

## Workflow
1. Data Import
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Feature Importance
6. Model Building (Regression)
7. Hyperparameter Tuning
8. Streamlit App Creation

## Dataset
- **Type:** Multivariate, Regression
- **Domain:** Biology
- **Instances:** 45,730
- **Features:** Real-valued (see data dictionary below)

### Data Dictionary
- RMSD: Size of the residue (target)
- F1: Total surface area
- F2: Non polar exposed area
- F3: Fractional area of exposed non polar residue
- F4: Fractional area of exposed non polar part of residue
- F5: Molecular mass weighted exposed area
- F6: Average deviation from standard exposed area of residue
- F7: Euclidian distance
- F8: Secondary structure penalty
- F9: Spacial Distribution constraints (N,K Value)

## Folder Structure
- `data/` - Raw data files
- `notebooks/` - Jupyter notebooks
- `src/` - Source code scripts
- `models/` - Saved models
- `outputs/` - Results and outputs

---

## Getting Started
1. Create and activate the virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the workflow scripts in `src/` or use the provided notebooks.