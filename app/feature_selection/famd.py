import pandas as pd
import math
from sklearn.decomposition import PCA
import numpy as np


def calculate_zscore(df, columns):
    """ scales columns in dataframe using z-score """
    df = df.copy()
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df


def one_hot_encode(df, columns):
    """
    one hot encodes list of columns and
    concatenates them to the original df
    """
    concat_df = pd.concat([pd.get_dummies(df[col], drop_first=True, prefix=col) for col in columns], axis=1)
    one_hot_cols = concat_df.columns
    return concat_df, one_hot_cols


def normalize_column_modality(df, columns):
    """divides each column by the probability μₘ of the modality
    (number of ones in the column divided by N) only for one hot columns
    """
    length = len(df)
    for col in columns:
        weight = math.sqrt(sum(df[col]) / length)
        df[col] = df[col] / weight
    return df


def center_columns(df, columns):
    """center columns by subtracting the mean value"""
    for col in columns:
        df[col] = (df[col] - df[col].mean())
    return df


def FAMD_(df, n_components=2):
    """
    Factorial Analysis of Mixed Data (FAMD),
    which generalizes the Principal Component Analysis (PCA)
    algorithm to datasets containing numerical and categorical variables

    a) For the numerical variables
      - Standard scale (= get the z-score)

    b) For the categorical variables:
      - Get the one-hot encoded columns
      - Divide each column by the square root of its probability sqrt(μₘ)
      - Center the columns

    c) Apply a PCA algorithm over the table obtained!
    """

    variable_distances = []

    numeric_cols = df.select_dtypes(include=np.number)
    cat_cols = df.select_dtypes(include='object')

    # numeric process
    normalized_df = calculate_zscore(df, numeric_cols)
    normalized_df = normalized_df[numeric_cols.columns]

    # categorical process
    cat_one_hot_df, one_hot_cols = one_hot_encode(df, cat_cols)
    cat_one_hot_norm_df = normalize_column_modality(cat_one_hot_df, one_hot_cols)
    cat_one_hot_norm_center_df = center_columns(cat_one_hot_norm_df, one_hot_cols)

    # Merge DataFrames
    processed_df = pd.concat([normalized_df, cat_one_hot_norm_center_df], axis=1)

    # Perform (PCA)
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(processed_df)
    return result
