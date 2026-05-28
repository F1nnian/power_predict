# power_predict

Group Project "Power Predict" | PS Machine Learning UIBK

Click [here](https://www.overleaf.com/project/6a1863f7475e18bcff2dd52a) to read the report.

## Getting Started & How to Run (AI-generated)

Follow these steps to set up your local environment, prepare the dataset, and run the Power Predict machine learning pipeline.

### 1. Prerequisites

Make sure you have Python 3.10 or higher installed on your machine.

### 2. Environment Setup

Clone the repository, create a virtual environment, and install the required dependencies:

```bash
# Clone the repository
git clone [https://github.com/F1nnian/power_predict.git](https://github.com/F1nnian/power_predict.git)
cd power_predict

# Create a virtual environment (venv)
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Install required Python packages
pip install -r requirements.txt
```

### 3. Running the Pipeline

```bash
# Step 1: Preprocess the data (handles missing values, scaling, etc.)
python src/preprocessing.py

# Step 2: Train the final ML model
python src/train.py

# Step 3: Evaluate the model and generate report plots
python src/evaluate.py
```

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
