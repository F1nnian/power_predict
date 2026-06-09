import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "power_consumption"
RANDOM_STATE = 42


def find_dataset_path():
    """
    Finds the dataset file in the most common project locations.
    """

    possible_paths = [
        os.path.join("data", "powerpredict.csv"),
        os.path.join("data", "powerpredict.csv.zip"),
        os.path.join("data", "raw", "powerpredict.csv"),
        os.path.join("data", "raw", "powerpredict.csv.zip"),
        "powerpredict.csv",
        "powerpredict.csv.zip",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Dataset not found. Please put powerpredict.csv or powerpredict.csv.zip "
        "inside the data/ folder."
    )


def load_dataset():
    """
    Loads the Power Predict dataset.
    """

    dataset_path = find_dataset_path()
    df = pd.read_csv(dataset_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    return df


def split_features_target(df):
    """
    Separates input features X from the target variable y.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def create_train_validation_split(X, y, test_size=0.2):
    """
    Splits the data into training and validation sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE
    )


def get_feature_types(X):
    """
    Detects numerical and categorical columns.
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X):
    """
    Builds the preprocessing pipeline.

    Numerical columns:
    - missing values are replaced with the median
    - values are standardized

    Categorical columns:
    - missing values are replaced with the most frequent value
    - categories are converted into numbers using OneHotEncoder
    """

    numeric_features, categorical_features = get_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor