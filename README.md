# power_predict

Group Project "Power Predict" | PS Machine Learning UIBK

## File structure (AI-generated)

```
power-predict/
│
├── .gitignore               # Specifies intentionally untracked files to ignore (e.g., large data files)
├── README.md                # Project overview, setup instructions, and team member info
├── requirements.txt         # Python dependencies (e.g., scikit-learn, pandas, matplotlib)
│
├── data/                    # Dataset storage (Do not push massive datasets to GitHub!)
│   ├── raw/                 # Original, unaltered data
│   └── processed/           # Data after preprocessing (handling missing values, scaling, etc.)
│
├── notebooks/               # Jupyter Notebooks for exploratory data analysis (EDA) and prototyping
│   ├── 01_eda.ipynb         # Analyzing features, data imbalances, and missing data
│   └── 02_experiments.ipynb # Initial model testing and hyperparameter tuning
│
├── src/                     # Production-ready Python source code
│   ├── __init__.py
│   ├── preprocessing.py     # Functions for feature extraction, scaling, encoding
│   ├── train.py             # Script to train the chosen ML algorithm
│   └── evaluate.py          # Script to compute metrics and generate report visualizations
│
├── models/                  # Saved model weights, checkpoints, or serialized pipelines (e.g., .pkl)
│   └── final_model.pkl
│
└── report/                  # Overleaf / LaTeX Project Folder
    ├── main.tex             # Main LaTeX document containing your sections (I-V)
    ├── references.bib       # Bibliography/References file
    └── figures/             # Visualizations required for Section III & IV
        ├── metric_plots.png
        └── model_structure.png
```
